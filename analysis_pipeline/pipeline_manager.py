#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Manager for Kubernetes Noisy Neighbours Lab
This module provides a high-level API for orchestrating the analysis pipeline 
across metrics, phases, and tenants.
"""

import logging
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
from analysis_pipeline.causal_fixed import CausalAnalysisFixed

class PipelineManager:
    """
    Manages the execution and integration of different analysis modules in the pipeline.
    """
    
    def __init__(self, data_loader, output_dir):
        """
        Initialize the pipeline manager.
        
        Args:
            data_loader: Initialized DataLoader instance with loaded data
            output_dir: Base output directory for analysis results
        """
        self.data_loader = data_loader
        self.output_dir = Path(output_dir) if output_dir else None
        
        # Track analysis status
        self.execution_status = {
            "metrics_analysis": False,
            "phase_analysis": False,
            "tenant_analysis": False,
            "advanced_analysis": False,
            "suggestions": False,
            "causal_analysis": False
        }
        
        # Track execution time
        self.execution_times = {}
        
        logging.info(f"Initialized PipelineManager with output directory: {self.output_dir}")
    
    def run_pipeline(self, analyzers, args):
        """
        Execute the full analysis pipeline.
        
        Args:
            analyzers: Dictionary of analyzer instances {metrics, phase, tenant, suggestion, causal}
            args: Command line arguments
        
        Returns:
            dict: Execution status and summary
        """
        # Start timer for full pipeline
        start_time = datetime.now()
        
        pipeline_results = {}
        
        # Execute each analysis component if requested
        if args.metrics_analysis:
            pipeline_results["metrics_analysis"] = self.run_metrics_analysis(
                analyzers["metrics"], args
            )
        
        if args.phase_analysis:
            pipeline_results["phase_analysis"] = self.run_phase_analysis(
                analyzers["phase"], args
            )
        
        if args.tenant_analysis:
            pipeline_results["tenant_analysis"] = self.run_tenant_analysis(
                analyzers["tenant"], args
            )
            
        if args.causal_analysis:
            pipeline_results["causal_analysis"] = self.run_causal_analysis(
                analyzers["causal"], args
            )
        
        if args.suggest_visualizations:
            pipeline_results["suggestions"] = self.generate_suggestions(
                analyzers["metrics"], analyzers["suggestion"], args
            )
        
        # Calculate total execution time
        end_time = datetime.now()
        total_execution_time = (end_time - start_time).total_seconds()
        
        # Save execution summary
        summary = {
            "execution_time": total_execution_time,
            "components_executed": list(pipeline_results.keys()),
            "status": "Completed",
            "timestamp": end_time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if self.output_dir:
            with open(self.output_dir / "pipeline_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
        
        logging.info(f"Pipeline execution completed in {total_execution_time:.2f} seconds")
        return summary
    
    def run_metrics_analysis(self, metrics_analyzer, args):
        """
        Execute metrics-focused analysis.
        
        Args:
            metrics_analyzer: MetricsAnalyzer instance
            args: Command line arguments
        
        Returns:
            dict: Results summary
        """
        logging.info("Starting metrics-focused analysis...")
        start_time = datetime.now()
        
        # Process each component and phase
        results = {}
        
        # Get all metrics for each phase
        all_phase_metrics = {}
        for phase in args.phases:
            # Since get_all_metrics_for_phase doesn't exist, we'll use the data structure directly
            if phase in self.data_loader.data:
                all_phase_metrics[phase] = self.data_loader.data[phase]
        
        # Process each component
        for component in args.components:
            logging.info(f"Analyzing metrics for component: {component}")
            component_results = {}
            
            # Get metrics for this component across all phases
            for phase in args.phases:
                # Check if phase has data
                if phase in all_phase_metrics:
                    # Get component metrics from this phase
                    comp_metrics = self.data_loader.get_all_metrics_for_component(phase, component)
                    
                    if comp_metrics:
                        # Run basic analysis for each metric
                        for metric_name, metric_data in comp_metrics.items():
                            if metric_data is not None and not metric_data.empty:
                                # Run statistical analysis
                                stats = metrics_analyzer.calculate_statistics(metric_data, f"{component}_{metric_name}_{phase}")
                                
                                if stats is not None:
                                    if phase not in component_results:
                                        component_results[phase] = {}
                                    
                                    component_results[phase][metric_name] = {
                                        "stats": stats
                                    }
                                    
                                    # Run time series analysis if not skipping
                                    if not args.skip_advanced:
                                        ts_results = metrics_analyzer.analyze_timeseries(metric_data, f"{component}_{metric_name}_{phase}")
                                        component_results[phase][metric_name]["timeseries"] = ts_results
                                    
                                    # Create plots if not skipping
                                    if not args.skip_plots:
                                        metrics_analyzer.create_plots(metric_data, f"{component}_{metric_name}_{phase}")
            
            # Add component results to overall results
            if component_results:
                results[component] = component_results
            
            # Cross-phase metric analysis
            if len(args.phases) >= 2:
                cross_phase_results = {}
                
                # Find common metrics across phases
                common_metrics = self._find_common_metrics(component, args.phases)
                
                for metric in common_metrics:
                    # Get this metric across all phases
                    metrics_across_phases = {}
                    
                    for phase in args.phases:
                        comp_metrics = self.data_loader.get_all_metrics_for_component(phase, component)
                        if comp_metrics and metric in comp_metrics:
                            metrics_across_phases[phase] = comp_metrics[metric]
                    
                    # Create data series dictionary for correlation analysis
                    series_dict = {}
                    for phase, data in metrics_across_phases.items():
                        if isinstance(data, pd.DataFrame) and not data.empty:
                            # Get first numeric column
                            value_col = None
                            for col in data.columns:
                                if pd.api.types.is_numeric_dtype(data[col]):
                                    value_col = col
                                    break
                                    
                            if value_col:
                                series_dict[phase] = data[value_col]
                    
                    # Correlation analysis between phases
                    if len(series_dict) >= 2 and not args.skip_advanced:
                        cross_phase_results[metric] = metrics_analyzer.analyze_correlations(
                            series_dict,
                            title=f"{component}_{metric}_across_phases",
                            corr_method='pearson'
                        )
                
                # Add cross-phase results
                if cross_phase_results:
                    if "cross_phase" not in results:
                        results["cross_phase"] = {}
                    results["cross_phase"][component] = cross_phase_results
        
        # Record execution time
        end_time = datetime.now()
        self.execution_times["metrics_analysis"] = (end_time - start_time).total_seconds()
        self.execution_status["metrics_analysis"] = True
        
        logging.info("Metrics-focused analysis complete")
        return results
    
    def run_phase_analysis(self, phase_analyzer, args):
        """
        Execute phase comparison analysis.
        
        Args:
            phase_analyzer: PhaseAnalyzer instance
            args: Command line arguments
            
        Returns:
            dict: Results summary
        """
        logging.info("Starting phase comparison analysis...")
        start_time = datetime.now()
        
        results = {}
        
        # Get all metrics for each phase
        all_phase_metrics = {}
        for phase in args.phases:
            # Since get_all_metrics_for_phase doesn't exist, we'll use the data structure directly
            if phase in self.data_loader.data:
                all_phase_metrics[phase] = self.data_loader.data[phase]
        
        # Get specified metrics of interest or select key metrics
        metrics_to_analyze = args.metrics_of_interest if args.metrics_of_interest else [
            'cpu_usage', 'memory_usage', 'network_total_bandwidth', 'disk_io_total'
        ]
        
        # Compare phases for each component and metric
        for component in args.components:
            logging.info(f"Analyzing phase differences for component: {component}")
            component_results = {}
            
            for metric in metrics_to_analyze:
                # Get this metric across all phases
                metric_by_phase = {}
                
                for phase in args.phases:
                    comp_metrics = self.data_loader.get_all_metrics_for_component(phase, component)
                    if comp_metrics and metric in comp_metrics:
                        metric_by_phase[phase] = comp_metrics[metric]
                
                # If we have data from at least 2 phases, compare them
                if len(metric_by_phase) >= 2:
                    # Run phase comparison analysis
                    phase_comparison = phase_analyzer.compare_phases(
                        metric_by_phase, 
                        metric_name=f"{component} {metric}",
                        phase_names=list(metric_by_phase.keys()),
                        comparison_methods=['boxplot', 'violin', 'stats_test']
                    )
                    
                    if phase_comparison:
                        if metric not in component_results:
                            component_results[metric] = {}
                        component_results[metric] = phase_comparison
                        
                    # If change point detection is enabled
                    if args.change_point_detection and not args.skip_advanced:
                        # Focus on one phase at a time (typically attack phase)
                        attack_phase = next((p for p in args.phases if "attack" in p.lower()), None)
                        if attack_phase and attack_phase in metric_by_phase:
                            change_points = phase_analyzer.detect_change_points(
                                metric_by_phase[attack_phase],
                                f"{component} {metric} ({attack_phase})"
                            )
                            
                            if change_points:
                                if "change_points" not in component_results:
                                    component_results["change_points"] = {}
                                component_results["change_points"][metric] = {attack_phase: change_points}
                
                # Run recovery analysis if requested
                if args.recovery_analysis and len(args.phases) >= 3:
                    # Find baseline, attack and recovery phases
                    baseline = next((p for p in args.phases if "baseline" in p.lower()), None)
                    attack = next((p for p in args.phases if "attack" in p.lower()), None)
                    recovery = next((p for p in args.phases if "recovery" in p.lower()), None)
                    
                    if baseline and attack and recovery:
                        if all(phase in metric_by_phase for phase in [baseline, attack, recovery]):
                            recovery_analysis = phase_analyzer.analyze_recovery(
                                metric_by_phase,
                                baseline_phase=baseline,
                                attack_phase=attack,
                                recovery_phase=recovery,
                                metric_name=f"{component} {metric}"
                            )
                            
                            if recovery_analysis:
                                if "recovery_analysis" not in component_results:
                                    component_results["recovery_analysis"] = {}
                                component_results["recovery_analysis"][metric] = recovery_analysis
            
            # Add component results to overall results
            if component_results:
                results[component] = component_results
        
        # Record execution time
        end_time = datetime.now()
        self.execution_times["phase_analysis"] = (end_time - start_time).total_seconds()
        self.execution_status["phase_analysis"] = True
        
        logging.info("Phase comparison analysis complete")
        return results
    
    def run_tenant_analysis(self, tenant_analyzer, args):
        """
        Execute tenant-focused analysis.
        
        Args:
            tenant_analyzer: TenantAnalyzer instance
            args: Command line arguments
            
        Returns:
            dict: Results summary
        """
        logging.info("Starting tenant-focused analysis...")
        start_time = datetime.now()
        
        results = {}
        
        # Get metrics of interest
        metrics_to_analyze = args.metrics_of_interest if args.metrics_of_interest else [
            'cpu_usage', 'memory_usage', 'network_total_bandwidth', 'disk_io_total'
        ]
        
        # Compare tenants for each phase and metric
        for phase in args.phases:
            logging.info(f"Analyzing tenant differences for phase: {phase}")
            phase_results = {}
            
            for metric in metrics_to_analyze:
                # Get this metric across all tenants
                tenants_data = {}
                
                for tenant in args.components:
                    # Skip non-tenant components
                    if not tenant.startswith("tenant-"):
                        continue
                        
                    comp_metrics = self.data_loader.get_all_metrics_for_component(phase, tenant)
                    if comp_metrics and metric in comp_metrics:
                        tenants_data[tenant] = comp_metrics[metric]
                
                # If we have data from at least 2 tenants, compare them
                if len(tenants_data) >= 2:
                    # Run tenant comparison analysis
                    comparison = tenant_analyzer.compare_tenants(
                        tenants_data,
                        metric_name=metric,
                        phase=phase
                    )
                    
                    if comparison is not None:
                        if metric not in phase_results:
                            phase_results[metric] = {}
                        phase_results[metric]["comparison"] = comparison
                        
                    # Run noisy neighbor detection if enough tenants
                    if len(tenants_data) >= 3 and not args.skip_advanced:
                        neighbor_analysis = tenant_analyzer.identify_noisy_neighbors(
                            tenants_data,
                            metric_name=metric,
                            phase=phase
                        )
                        
                        if neighbor_analysis:
                            if metric not in phase_results:
                                phase_results[metric] = {}
                            phase_results[metric]["noisy_neighbors"] = neighbor_analysis
                
                # If tenant degradation analysis is requested
                if args.tenant_degradation and len(tenants_data) >= 2:
                    degradation = tenant_analyzer.analyze_tenant_degradation(
                        tenants_data,
                        metric_name=metric,
                        phase=phase
                    )
                    
                    if degradation:
                        if metric not in phase_results:
                            phase_results[metric] = {}
                        phase_results[metric]["degradation"] = degradation
            
            # Add phase results to overall results
            if phase_results:
                results[phase] = phase_results
        
        # Generate cross-tenant analysis if requested
        if args.tenant_comparison:
            cross_tenant_results = {}
            
            # Get all tenant metrics across phases
            for metric in metrics_to_analyze:
                metric_results = {}
                
                # For each tenant
                for tenant in args.components:
                    if not tenant.startswith("tenant-"):
                        continue
                        
                    # Get metrics for this tenant across all phases
                    tenant_across_phases = {}
                    
                    for phase in args.phases:
                        comp_metrics = self.data_loader.get_all_metrics_for_component(phase, tenant)
                        if comp_metrics and metric in comp_metrics:
                            tenant_across_phases[phase] = comp_metrics[metric]
                    
                    # If we have data from multiple phases, analyze
                    if len(tenant_across_phases) >= 2:
                        result = tenant_analyzer.compare_tenant_across_phases(
                            tenant_across_phases,
                            tenant_name=tenant,
                            metric_name=metric
                        )
                        
                        if result:
                            if tenant not in metric_results:
                                metric_results[tenant] = {}
                            metric_results[tenant]["cross_phase"] = result
                
                # Add to results if we found any
                if metric_results:
                    cross_tenant_results[metric] = metric_results
            
            # Add to overall results
            if cross_tenant_results:
                results["cross_tenant_analysis"] = cross_tenant_results
        
        # Record execution time
        end_time = datetime.now()
        self.execution_times["tenant_analysis"] = (end_time - start_time).total_seconds()
        self.execution_status["tenant_analysis"] = True
        
        logging.info("Tenant-focused analysis complete")
        return results

    def generate_suggestions(self, metrics_analyzer, suggestion_engine, args):
        """
        Generate suggestions for visualizations and analysis.
        
        Args:
            metrics_analyzer: MetricsAnalyzer instance
            suggestion_engine: SuggestionEngine instance
            args: Command line arguments
            
        Returns:
            dict: Suggestions
        """
        logging.info("Generating analysis and visualization suggestions...")
        start_time = datetime.now()
        
        suggestions = {}
        
        # Get all phase metrics
        if args.phases:
            first_phase = args.phases[0]
            all_components = {}
            
            for component in args.components:
                comp_metrics = self.data_loader.get_all_metrics_for_component(first_phase, component)
                if comp_metrics:
                    all_components[component] = comp_metrics
            
            # Get suggestions for each component's metrics
            for component, metrics in all_components.items():
                component_suggestions = {}
                
                # Limit to first 5 metrics
                for i, (metric_name, metric_data) in enumerate(metrics.items()):
                    if i >= 5:
                        break
                        
                    # Get plot suggestions
                    plot_suggestions = metrics_analyzer.suggest_plots(metric_data, f"{component}_{metric_name}")
                    
                    if plot_suggestions and plot_suggestions != {"error": "Dados insuficientes para sugestões"}:
                        # Save suggestions
                        if "visualization" not in component_suggestions:
                            component_suggestions["visualization"] = {}
                        component_suggestions["visualization"][metric_name] = plot_suggestions
                        
                        # Also save to file
                        if metrics_analyzer.output_dir:
                            with open(metrics_analyzer.output_dir / f"{component}_{metric_name}_suggestions.txt", 'w') as f:
                                f.write(f"Visualization Suggestions for {component} - {metric_name}:\n\n")
                                
                                for viz_key, viz_data in plot_suggestions.items():
                                    if viz_key != "error":
                                        f.write(f"- {viz_data['type']}: {viz_data['justification']}\n")
                    
                    # Get table suggestions
                    table_suggestions = metrics_analyzer.suggest_tables(metric_data, f"{component}_{metric_name}")
                    
                    if table_suggestions and table_suggestions != {"error": "Dados insuficientes para sugestões"}:
                        if "tables" not in component_suggestions:
                            component_suggestions["tables"] = {}
                        component_suggestions["tables"][metric_name] = table_suggestions
                        
                        # Also save to file
                        if metrics_analyzer.output_dir:
                            with open(metrics_analyzer.output_dir / f"{component}_{metric_name}_table_suggestions.txt", 'w') as f:
                                f.write(f"Table Suggestions for {component} - {metric_name}:\n\n")
                                
                                for table_key, table_data in table_suggestions.items():
                                    if table_key != "error":
                                        formats = ", ".join(table_data['format']) if 'format' in table_data else "CSV"
                                        f.write(f"- {table_data['type']} ({formats}): {table_data['justification']}\n")
                
                # Add to overall suggestions
                if component_suggestions:
                    suggestions[component] = component_suggestions
        
        # Record execution time
        end_time = datetime.now()
        self.execution_times["suggestions"] = (end_time - start_time).total_seconds()
        self.execution_status["suggestions"] = True
        
        logging.info("Suggestions generation complete")
        return suggestions
    
    def run_causal_analysis(self, causal_analyzer, args):
        """
        Execute causal analysis between metrics.
        
        Args:
            causal_analyzer: CausalAnalysisFixed instance
            args: Command line arguments
            
        Returns:
            dict: Results summary
        """
        logging.info("Starting causal analysis...")
        start_time = datetime.now()
        
        results = {}
        
        # Get specified metrics of interest or select key metrics
        metrics_to_analyze = args.metrics_of_interest if hasattr(args, 'metrics_of_interest') else [
            'cpu_usage', 'memory_usage', 'network_total_bandwidth', 'disk_io_total'
        ]
        
        # Get specified causal method or use default
        causal_method = args.causal_method if hasattr(args, 'causal_method') else 'toda-yamamoto'
        
        # Run causal analysis on the data
        for phase in args.phases:
            if phase in self.data_loader.data:
                phase_data = {phase: self.data_loader.data[phase]}
                
                # Run the analysis
                causal_results = causal_analyzer.run_causal_analysis(
                    phase_data=phase_data,
                    method=causal_method,
                    metrics_of_interest=metrics_to_analyze,
                    components=args.components,
                    save_results=True
                )
                
                if not causal_results.empty:
                    # If we have results, add to the results dict
                    if phase not in results:
                        results[phase] = {}
                    
                    # Convert DataFrame to dict for JSON serialization
                    results[phase][causal_method] = causal_results.to_dict(orient='records')
        
        # Record execution time
        end_time = datetime.now()
        self.execution_times["causal_analysis"] = (end_time - start_time).total_seconds()
        self.execution_status["causal_analysis"] = True
        
        logging.info("Causal analysis complete")
        return results
    
    def _find_common_metrics(self, component, phases):
        """
        Find metrics that are common across all specified phases for a component.
        
        Args:
            component (str): Component name
            phases (list): List of phases
            
        Returns:
            list: List of metrics available in all phases
        """
        common_metrics = None
        
        for phase in phases:
            comp_metrics = self.data_loader.get_all_metrics_for_component(phase, component)
            if comp_metrics:
                metrics_set = set(comp_metrics.keys())
                
                if common_metrics is None:
                    common_metrics = metrics_set
                else:
                    common_metrics = common_metrics.intersection(metrics_set)
        
        return list(common_metrics) if common_metrics else []
