import unittest
import numpy as np
import pandas as pd
import networkx as nx
import sys
import os

# Adjust path to import modules from the project root
sys.path.insert(0, '/home/phil/Projects/k8s-noisy-detection')

from analysis_modules.root_cause import RootCauseAnalyzer

class TestRootCauseAnalyzer(unittest.TestCase):

    def setUp(self):
        self.tenant_names = ['T1', 'T2', 'T3', 'T4', 'T5']
        self.impact_matrix = np.array([
            # T1   T2   T3   T4   T5
            [0.0, 0.9, 0.1, 0.0, 0.0],  # T1 impacts T2 (strong), T3 (weak - below network threshold)
            [0.0, 0.0, 0.8, 0.0, 0.0],  # T2 impacts T3 (strong)
            [0.0, 0.0, 0.0, 0.7, 0.0],  # T3 impacts T4 (strong)
            [0.0, 0.0, 0.0, 0.0, 0.6],  # T4 impacts T5 (strong)
            [0.4, 0.0, 0.0, 0.0, 0.0]   # T5 impacts T1 (moderate)
        ])
        # metrics_data is not deeply used by the core logic of RootCauseAnalyzer methods being tested here,
        # so a placeholder is often sufficient for these unit tests.
        self.metrics_data = {} 
        
        self.analyzer = RootCauseAnalyzer(
            impact_matrix=self.impact_matrix,
            metrics_data=self.metrics_data,
            tenant_names=self.tenant_names
        )

    def test_build_impact_network(self):
        network = self.analyzer._build_impact_network()
        self.assertIsInstance(network, nx.DiGraph)
        self.assertCountEqual(network.nodes(), self.tenant_names)
        
        expected_edges = [('T1', 'T2'), ('T2', 'T3'), ('T3', 'T4'), ('T4', 'T5'), ('T5', 'T1')]
        # Check edges based on the threshold (0.3) in _build_impact_network
        present_edges = [edge for edge in network.edges() if network.edges[edge]['weight'] > 0.3]
        self.assertCountEqual(present_edges, expected_edges)
        self.assertAlmostEqual(network.edges[('T1', 'T2')]['weight'], 0.9)
        self.assertAlmostEqual(network.edges[('T5', 'T1')]['weight'], 0.4)
        # Edge T1->T3 should not exist due to weight 0.1 < 0.3 threshold
        self.assertFalse(network.has_edge('T1', 'T3'))


    def test_calculate_path_strength(self):
        # Path: T1 -> T2 -> T3
        path1 = ['T1', 'T2', 'T3']
        # Strength = impact(T1,T2) * impact(T2,T3) = 0.9 * 0.8 = 0.72
        strength1 = self.analyzer._calculate_path_strength(path1)
        self.assertAlmostEqual(strength1, 0.9 * 0.8)

        # Path: T1 -> T2 -> T3 -> T4
        path2 = ['T1', 'T2', 'T3', 'T4']
        # Strength = 0.9 * 0.8 * 0.7 = 0.504
        strength2 = self.analyzer._calculate_path_strength(path2)
        self.assertAlmostEqual(strength2, 0.9 * 0.8 * 0.7)
        
        # Path with a weak link that might be filtered by network creation but path calc should still work
        # If T1->T3 (0.1) was allowed in a path for some reason:
        # For this test, we directly use impact_matrix, not the filtered network
        # Create a temporary path that wouldn't form in the network due to threshold
        # but the calculation logic itself should be sound.
        # Let's test a path that is valid according to the matrix: T5 -> T1 -> T2
        path3 = ['T5', 'T1', 'T2']
        strength3 = self.analyzer._calculate_path_strength(path3)
        self.assertAlmostEqual(strength3, 0.4 * 0.9)

        # Single node path
        self.assertEqual(self.analyzer._calculate_path_strength(['T1']), 0)
        # Empty path
        self.assertEqual(self.analyzer._calculate_path_strength([]), 0)

    def test_first_order_analysis(self):
        results = self.analyzer._first_order_analysis()
        # T1 impacts T2 (0.9 > 0.5)
        self.assertIn('T1', results)
        self.assertEqual(len(results['T1']['victims']), 1)
        self.assertEqual(results['T1']['victims'][0]['victim'], 'T2')
        self.assertAlmostEqual(results['T1']['victims'][0]['impact_strength'], 0.9)
        self.assertAlmostEqual(results['T1']['avg_impact'], 0.9)

        # T2 impacts T3 (0.8 > 0.5)
        self.assertIn('T2', results)
        self.assertEqual(results['T2']['victims'][0]['victim'], 'T3')

        # T5 impacts T1 (0.4, not > 0.5)
        self.assertNotIn('T5', results) # T5's impact on T1 is 0.4, which is not > 0.5

    def test_second_order_analysis(self):
        results = self.analyzer._second_order_analysis()
        # For T1: T1 -> T2 (0.9) -> T3 (0.8). Cascade strength = 0.9 * 0.8 = 0.72
        self.assertIn('T1', results)
        self.assertEqual(len(results['T1']['cascading_effects']), 1)
        cascade_effect = results['T1']['cascading_effects'][0]
        self.assertEqual(cascade_effect['secondary_victim'], 'T3')
        self.assertEqual(cascade_effect['intermediate_victim'], 'T2')
        self.assertAlmostEqual(cascade_effect['cascade_strength'], 0.9 * 0.8)
        self.assertEqual(results['T1']['total_secondary_victims'], 1)

        # For T5: T5 -> T1 (0.4 - direct impact too weak for 1st order, so no 2nd order from T5 via T1)
        # The first link in a cascade must be > 0.5 (direct victim)
        # The second link must be > 0.3
        # T5 impacts T1 with 0.4, which is not > 0.5, so T1 is not a direct victim of T5 for this analysis.
        self.assertNotIn('T5', results)


    def test_higher_order_analysis_order_3(self):
        # Test for order 3: T1 -> T2 -> T3 -> T4
        # Path strength = 0.9 * 0.8 * 0.7 = 0.504
        # This strength (0.504) should be > 0.1 (threshold in _higher_order_analysis)
        results = self.analyzer._higher_order_analysis(max_order=3)
        self.assertIn('order_3', results)
        order_3_impacts = results['order_3']
        
        self.assertIn('T1', order_3_impacts)
        t1_higher_order = order_3_impacts['T1']
        self.assertEqual(len(t1_higher_order['higher_order_paths']), 1)
        path_data = t1_higher_order['higher_order_paths'][0]
        self.assertEqual(path_data['target'], 'T4')
        self.assertEqual(path_data['path'], 'T1 → T2 → T3 → T4')
        self.assertAlmostEqual(path_data['path_strength'], 0.9 * 0.8 * 0.7)
        self.assertEqual(path_data['order'], 3)

    def test_higher_order_analysis_max_order_2(self):
        # If max_order is 2, order_3 should not be present
        results = self.analyzer._higher_order_analysis(max_order=2)
        self.assertNotIn('order_3', results)
        self.assertNotIn('order_4', results)

    def test_aggregate_multi_order_results(self):
        # Mock results from other analyses
        mock_analysis_results = {
            'first_order': self.analyzer._first_order_analysis(),
            'second_order': self.analyzer._second_order_analysis(),
            'higher_order': self.analyzer._higher_order_analysis(max_order=3)
        }
        
        aggregated = self.analyzer._aggregate_multi_order_results(mock_analysis_results)
        self.assertIn('T1', aggregated)
        t1_agg = aggregated['T1']

        # Expected for T1:
        # First order: T1 -> T2 (impact 0.9, 1 victim) -> score part = 0.9 * 1 = 0.9
        # Second order: T1 -> T2 -> T3 (cascade 0.72) -> score part = 0.72 * 0.7 = 0.504
        # Higher order (3): T1 -> T2 -> T3 -> T4 (system_wide_impact for T1 in order_3 is 0.504)
        #   weight for order 3 is 0.5**(3-2) = 0.5. Score part = 0.504 * 0.5 = 0.252
        # Total score = 0.9 + 0.504 + 0.252 = 1.656
        
        expected_t1_score = (0.9 * 1) + (0.72 * 0.7) + (0.504 * 0.5)
        self.assertAlmostEqual(t1_agg['total_impact_score'], expected_t1_score)
        self.assertCountEqual(t1_agg['orders_involved'], [1, 2, 3])
        self.assertEqual(t1_agg['victim_counts']['direct'], 1)
        self.assertEqual(t1_agg['victim_counts']['cascading'], 1) # T3 is secondary victim
        self.assertEqual(t1_agg['victim_counts']['higher_order'], 1) # T4 is target
        self.assertIn('direct_impact', t1_agg['impact_patterns'])
        self.assertIn('cascading_effect', t1_agg['impact_patterns'])
        self.assertIn('system_wide_propagation', t1_agg['impact_patterns'])

    def test_calculate_multi_order_confidence(self):
        # Mock results
        mock_analysis_results = self.analyzer.analyze_all_orders(max_order=3) # Use the full analysis
        
        confidence_ranking = mock_analysis_results['confidence_ranking']
        
        self.assertIn('T1', confidence_ranking)
        t1_confidence = confidence_ranking['T1']
        
        # Check some basic properties of confidence score
        self.assertGreaterEqual(t1_confidence['overall_confidence'], 0)
        self.assertLessEqual(t1_confidence['overall_confidence'], 1)
        self.assertIn('normalized_impact', t1_confidence)
        self.assertIn('order_diversity', t1_confidence)
        self.assertIn('pattern_diversity', t1_confidence)

        # Check if sorted (T1 should likely be high due to multiple orders of impact)
        if len(confidence_ranking) > 1:
            scores = [data['overall_confidence'] for data in confidence_ranking.values()]
            self.assertTrue(all(scores[i] >= scores[i+1] for i in range(len(scores)-1)))


    def test_analyze_all_orders(self):
        results = self.analyzer.analyze_all_orders(max_order=3)
        
        self.assertIn('first_order', results)
        self.assertIn('second_order', results)
        self.assertIn('higher_order', results)
        self.assertIn('order_3', results['higher_order'])
        self.assertIn('aggregated_results', results)
        self.assertIn('confidence_ranking', results)
        
        # Check if T1 is in aggregated results and confidence ranking
        self.assertIn('T1', results['aggregated_results'])
        self.assertIn('T1', results['confidence_ranking'])

if __name__ == '__main__':
    unittest.main()

