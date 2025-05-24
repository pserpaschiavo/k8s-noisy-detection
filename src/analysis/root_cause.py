"""
Root Cause Analysis Module
==========================

Implements multi-order analysis for noisy neighbor detection:
- First-order: Direct impact analysis
- Second-order: Cascading effects and indirect impacts
- Higher-order: System-wide propagation patterns

Dependencies: impact_matrix from existing modules
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from sklearn.cluster import DBSCAN
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class RootCauseAnalyzer:
    """
    Multi-order root cause analysis for noisy neighbor detection
    """
    
    def __init__(self, impact_matrix, metrics_data, tenant_names, phases=['baseline', 'attack', 'recovery']):
        self.impact_matrix = impact_matrix
        self.metrics_data = metrics_data
        self.tenant_names = tenant_names
        self.phases = phases
        self.n_tenants = len(tenant_names)
        
        # Build impact network for higher-order analysis
        self.impact_network = self._build_impact_network()
        
    def analyze_all_orders(self, max_order=3):
        """
        Perform complete multi-order root cause analysis
        """
        results = {
            'first_order': self._first_order_analysis(),
            'second_order': self._second_order_analysis(),
            'higher_order': self._higher_order_analysis(max_order),
            'aggregated_results': None,
            'confidence_ranking': None
        }
        
        # Aggregate results from all orders
        results['aggregated_results'] = self._aggregate_multi_order_results(results)
        results['confidence_ranking'] = self._calculate_multi_order_confidence(results)
        
        return results
    
    def _first_order_analysis(self):
        """
        First-order: Direct impact analysis
        """
        # Direct impacts from impact matrix
        direct_impacts = {}
        
        for i, suspect in enumerate(self.tenant_names):
            impacts_on_others = self.impact_matrix[i, :]
            victims = []
            
            for j, victim in enumerate(self.tenant_names):
                if i != j and impacts_on_others[j] > 0.5:  # Threshold for significant impact
                    victims.append({
                        'victim': victim,
                        'impact_strength': impacts_on_others[j],
                        'order': 1
                    })
            
            if victims:
                direct_impacts[suspect] = {
                    'victims': victims,
                    'total_victims': len(victims),
                    'avg_impact': np.mean([v['impact_strength'] for v in victims]),
                    'max_impact': np.max([v['impact_strength'] for v in victims])
                }
        
        return direct_impacts
    
    def _second_order_analysis(self):
        """
        Second-order: Cascading effects and indirect impacts
        """
        second_order_impacts = {}
        
        # For each potential root cause, find second-order victims
        for i, root_cause in enumerate(self.tenant_names):
            cascading_effects = []
            
            # Find direct victims of root cause
            direct_victims_indices = np.where(self.impact_matrix[i, :] > 0.5)[0]
            
            # For each direct victim, check who they impact (cascade)
            for victim_idx in direct_victims_indices:
                if victim_idx != i:  # Skip self
                    # Check who this victim impacts
                    secondary_victims_indices = np.where(self.impact_matrix[victim_idx, :] > 0.3)[0]
                    
                    for sec_victim_idx in secondary_victims_indices:
                        if sec_victim_idx != i and sec_victim_idx != victim_idx:  # Not root cause or direct victim
                            
                            # Calculate cascading impact strength
                            cascade_strength = (
                                self.impact_matrix[i, victim_idx] *  # Root → Direct victim
                                self.impact_matrix[victim_idx, sec_victim_idx]  # Direct → Secondary victim
                            )
                            
                            cascading_effects.append({
                                'secondary_victim': self.tenant_names[sec_victim_idx],
                                'intermediate_victim': self.tenant_names[victim_idx],
                                'cascade_strength': cascade_strength,
                                'path': f"{root_cause} → {self.tenant_names[victim_idx]} → {self.tenant_names[sec_victim_idx]}",
                                'order': 2
                            })
            
            if cascading_effects:
                second_order_impacts[root_cause] = {
                    'cascading_effects': cascading_effects,
                    'total_secondary_victims': len(set([e['secondary_victim'] for e in cascading_effects])),
                    'strongest_cascade': max(cascading_effects, key=lambda x: x['cascade_strength']),
                    'cascade_paths': [e['path'] for e in cascading_effects]
                }
        
        return second_order_impacts
    
    def _higher_order_analysis(self, max_order=3):
        """
        Higher-order: System-wide propagation using network analysis
        """
        higher_order_results = {}
        
        # Use NetworkX for path analysis
        G = self.impact_network
        
        for order in range(3, max_order + 1):
            order_impacts = {}
            
            for root_cause in self.tenant_names:
                if root_cause in G:
                    # Find all paths of specific length from root cause
                    paths_of_order = []
                    
                    for target in self.tenant_names:
                        if target != root_cause and target in G:
                            try:
                                # Find all simple paths of specific length
                                all_paths = list(nx.all_simple_paths(G, root_cause, target, cutoff=order))
                                paths_of_order_n = [path for path in all_paths if len(path) == order + 1]
                                
                                for path in paths_of_order_n:
                                    # Calculate cumulative impact through path
                                    path_strength = self._calculate_path_strength(path)
                                    
                                    if path_strength > 0.1:  # Threshold for significant higher-order impact
                                        paths_of_order.append({
                                            'target': target,
                                            'path': ' → '.join(path),
                                            'path_strength': path_strength,
                                            'order': order,
                                            'path_length': len(path) - 1
                                        })
                            except nx.NetworkXNoPath:
                                continue
                    
                    if paths_of_order:
                        order_impacts[root_cause] = {
                            'higher_order_paths': paths_of_order,
                            'total_targets': len(set([p['target'] for p in paths_of_order])),
                            'strongest_path': max(paths_of_order, key=lambda x: x['path_strength']),
                            'system_wide_impact': sum([p['path_strength'] for p in paths_of_order])
                        }
            
            higher_order_results[f'order_{order}'] = order_impacts
        
        return higher_order_results
    
    def _build_impact_network(self):
        """
        Build directed network from impact matrix
        """
        G = nx.DiGraph()
        
        # Add nodes
        for tenant in self.tenant_names:
            G.add_node(tenant)
        
        # Add edges based on impact matrix
        for i, source in enumerate(self.tenant_names):
            for j, target in enumerate(self.tenant_names):
                if i != j and self.impact_matrix[i, j] > 0.3:  # Threshold for edge creation
                    G.add_edge(source, target, weight=self.impact_matrix[i, j])
        
        return G
    
    def _calculate_path_strength(self, path):
        """
        Calculate cumulative impact strength through a path
        """
        if len(path) < 2:
            return 0
        
        cumulative_strength = 1.0
        
        for i in range(len(path) - 1):
            source_idx = self.tenant_names.index(path[i])
            target_idx = self.tenant_names.index(path[i + 1])
            edge_strength = self.impact_matrix[source_idx, target_idx]
            cumulative_strength *= edge_strength
        
        return cumulative_strength
    
    def _aggregate_multi_order_results(self, results):
        """
        Aggregate results from all orders to identify primary root causes
        """
        aggregated = {}
        
        for tenant in self.tenant_names:
            tenant_analysis = {
                'total_impact_score': 0,
                'orders_involved': [],
                'victim_counts': {'direct': 0, 'cascading': 0, 'higher_order': 0},
                'impact_patterns': []
            }
            
            # First order
            if tenant in results['first_order']:
                first_order = results['first_order'][tenant]
                tenant_analysis['total_impact_score'] += first_order['avg_impact'] * first_order['total_victims']
                tenant_analysis['orders_involved'].append(1)
                tenant_analysis['victim_counts']['direct'] = first_order['total_victims']
                tenant_analysis['impact_patterns'].append('direct_impact')
            
            # Second order
            if tenant in results['second_order']:
                second_order = results['second_order'][tenant]
                tenant_analysis['total_impact_score'] += second_order['strongest_cascade']['cascade_strength'] * 0.7  # Weight for second order
                tenant_analysis['orders_involved'].append(2)
                tenant_analysis['victim_counts']['cascading'] = second_order['total_secondary_victims']
                tenant_analysis['impact_patterns'].append('cascading_effect')
            
            # Higher order
            for order_key, order_data in results['higher_order'].items():
                if tenant in order_data:
                    higher_order = order_data[tenant]
                    weight = 0.5 ** (int(order_key.split('_')[1]) - 2)  # Decreasing weight for higher orders
                    tenant_analysis['total_impact_score'] += higher_order['system_wide_impact'] * weight
                    tenant_analysis['orders_involved'].append(int(order_key.split('_')[1]))
                    tenant_analysis['victim_counts']['higher_order'] += higher_order['total_targets']
                    tenant_analysis['impact_patterns'].append('system_wide_propagation')
            
            if tenant_analysis['orders_involved']:
                aggregated[tenant] = tenant_analysis
        
        return aggregated
    
    def _calculate_multi_order_confidence(self, results):
        """
        Calculate confidence scores for root cause identification
        """
        confidence_scores = {}
        aggregated = results['aggregated_results']
        
        if not aggregated:
            return confidence_scores
        
        # Normalize total impact scores
        max_impact = max([data['total_impact_score'] for data in aggregated.values()])
        
        for tenant, data in aggregated.items():
            # Factors for confidence:
            # 1. Normalized impact score
            # 2. Number of different orders involved
            # 3. Diversity of impact patterns
            # 4. Temporal consistency (if available)
            
            normalized_impact = data['total_impact_score'] / max_impact if max_impact > 0 else 0
            order_diversity = len(set(data['orders_involved'])) / 3  # Max 3 orders typically
            pattern_diversity = len(set(data['impact_patterns'])) / 3  # Max 3 patterns
            
            # Bonus for being involved in multiple orders
            multi_order_bonus = 0.2 if len(data['orders_involved']) > 1 else 0
            
            confidence = (
                0.5 * normalized_impact +
                0.2 * order_diversity +
                0.2 * pattern_diversity +
                0.1 * multi_order_bonus
            )
            
            confidence_scores[tenant] = {
                'overall_confidence': min(confidence, 1.0),
                'normalized_impact': normalized_impact,
                'order_diversity': order_diversity,
                'pattern_diversity': pattern_diversity,
                'orders_involved': data['orders_involved'],
                'impact_patterns': data['impact_patterns']
            }
        
        # Sort by confidence
        sorted_confidence = dict(sorted(confidence_scores.items(), 
                                      key=lambda x: x[1]['overall_confidence'], 
                                      reverse=True))
        
        return sorted_confidence

# Visualization functions
def plot_multi_order_analysis(rca_results, output_dir, filename_prefix):
    """
    Create comprehensive visualizations for multi-order analysis
    """
    # 1. Network visualization showing all orders
    plot_impact_network_multi_order(rca_results, output_dir, f"{filename_prefix}_network.png")
    
    # 2. Confidence ranking chart
    plot_confidence_ranking(rca_results['confidence_ranking'], output_dir, f"{filename_prefix}_confidence.png")
    
    # 3. Order comparison heatmap
    plot_order_comparison_heatmap(rca_results, output_dir, f"{filename_prefix}_orders.png")
    
    # 4. Impact propagation timeline
    plot_impact_propagation_timeline(rca_results, output_dir, f"{filename_prefix}_timeline.png")

def plot_impact_network_multi_order(rca_results, output_dir, filename):
    """
    Network visualization showing impacts across all orders
    """
    plt.figure(figsize=(14, 10))
    
    # Create network graph
    G = nx.DiGraph()
    
    # Add nodes and edges from all orders
    edge_colors = []
    edge_widths = []
    
    # First order (direct impacts) - Red edges
    if 'first_order' in rca_results:
        for root_cause, data in rca_results['first_order'].items():
            for victim_data in data['victims']:
                G.add_edge(root_cause, victim_data['victim'], 
                          weight=victim_data['impact_strength'], order=1)
                edge_colors.append('red')
                edge_widths.append(victim_data['impact_strength'] * 3)
    
    # Second order (cascading) - Orange edges
    if 'second_order' in rca_results:
        for root_cause, data in rca_results['second_order'].items():
            for cascade in data['cascading_effects']:
                path_nodes = cascade['path'].split(' → ')
                for i in range(len(path_nodes) - 1):
                    if not G.has_edge(path_nodes[i], path_nodes[i+1]):
                        G.add_edge(path_nodes[i], path_nodes[i+1], 
                                  weight=cascade['cascade_strength'], order=2)
                        edge_colors.append('orange')
                        edge_widths.append(cascade['cascade_strength'] * 3)
    
    # Higher order - Blue edges
    if 'higher_order' in rca_results:
        for order_key, order_data in rca_results['higher_order'].items():
            for root_cause, data in order_data.items():
                for path_data in data['higher_order_paths']:
                    path_nodes = path_data['path'].split(' → ')
                    for i in range(len(path_nodes) - 1):
                        if not G.has_edge(path_nodes[i], path_nodes[i+1]):
                            G.add_edge(path_nodes[i], path_nodes[i+1], 
                                      weight=path_data['path_strength'], 
                                      order=int(order_key.split('_')[1]))
                            edge_colors.append('blue')
                            edge_widths.append(path_data['path_strength'] * 3)
    
    # Layout and draw
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw nodes
    node_sizes = [len(list(G.predecessors(node))) * 300 + 500 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightgray', alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7, arrows=True)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Legend
    red_line = plt.Line2D([0], [0], color='red', linewidth=3, label='First Order (Direct)')
    orange_line = plt.Line2D([0], [0], color='orange', linewidth=3, label='Second Order (Cascading)')
    blue_line = plt.Line2D([0], [0], color='blue', linewidth=3, label='Higher Order (System-wide)')
    plt.legend(handles=[red_line, orange_line, blue_line], loc='upper right')
    
    plt.title("Multi-Order Impact Network Analysis", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence_ranking(confidence_ranking, output_dir, filename):
    """
    Bar chart showing confidence scores for root cause candidates
    """
    if not confidence_ranking:
        return
    
    tenants = list(confidence_ranking.keys())
    scores = [data['overall_confidence'] for data in confidence_ranking.values()]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(tenants, scores, color='steelblue', alpha=0.8)
    
    # Add confidence score labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Root Cause Confidence Ranking', fontsize=14, fontweight='bold')
    plt.xlabel('Tenant')
    plt.ylabel('Confidence Score')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1.1)
    
    # Color code: High confidence (>0.7) in red
    for i, score in enumerate(scores):
        if score > 0.7:
            bars[i].set_color('red')
            bars[i].set_alpha(0.8)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight')
    plt.close()

# Integration function
def perform_complete_root_cause_analysis(impact_matrix, metrics_data, tenant_names, output_dir):
    """
    Main function to perform complete multi-order root cause analysis
    """
    analyzer = RootCauseAnalyzer(impact_matrix, metrics_data, tenant_names)
    results = analyzer.analyze_all_orders(max_order=3)
    
    # Generate visualizations
    plot_multi_order_analysis(results, output_dir, "multi_order_rca")
    
    # Generate summary report
    generate_rca_summary_report(results, output_dir)
    
    return results

def generate_rca_summary_report(results, output_dir):
    """
    Generate text summary of root cause analysis
    """
    with open(f"{output_dir}/root_cause_analysis_report.txt", 'w') as f:
        f.write("ROOT CAUSE ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Top suspects
        if results['confidence_ranking']:
            f.write("TOP ROOT CAUSE SUSPECTS:\n")
            f.write("-" * 30 + "\n")
            for i, (tenant, data) in enumerate(list(results['confidence_ranking'].items())[:3]):
                f.write(f"{i+1}. {tenant}\n")
                f.write(f"   Confidence Score: {data['overall_confidence']:.3f}\n")
                f.write(f"   Orders Involved: {data['orders_involved']}\n")
                f.write(f"   Impact Patterns: {data['impact_patterns']}\n\n")
        
        # Multi-order summary
        f.write("MULTI-ORDER IMPACT SUMMARY:\n")
        f.write("-" * 30 + "\n")
        
        if 'first_order' in results:
            f.write(f"First-order impacts detected: {len(results['first_order'])}\n")
        
        if 'second_order' in results:
            f.write(f"Second-order cascades detected: {len(results['second_order'])}\n")
        
        if 'higher_order' in results:
            total_higher = sum(len(order_data) for order_data in results['higher_order'].values())
            f.write(f"Higher-order propagations detected: {total_higher}\n")