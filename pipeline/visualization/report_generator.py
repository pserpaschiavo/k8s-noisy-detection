"""
Generates comprehensive reports in various formats (Markdown, HTML, PDF).

This module consolidates analysis results, plots, and tables into
structured and human-readable reports.
"""

import os
import pandas as pd
from datetime import datetime
from pipeline.visualization.table_generator import convert_df_to_markdown

# Placeholder for a more sophisticated HTML/PDF generation library if needed
# from weasyprint import HTML # Example for PDF generation

def generate_markdown_report(experiment_results, config, output_dir, experiment_name="Experiment"):
    """
    Generates a Markdown report from the experiment results.

    Args:
        experiment_results (dict): A dictionary containing all results from the pipeline.
                                   Expected keys: 'processed_data', 'aggregated_data',
                                                  'phase_comparison_tables', 'impact_summary_tables',
                                                  'correlation_matrix_plots', 'entropy_results',
                                                  'causality_results', 'noisy_tenant_info',
                                                  'rounds_comparison_outputs', 'tenant_comparison_outputs',
                                                  'app_metrics_analysis_results',
                                                  'technology_comparison_results',
                                                  'comparison_experiment_results'.
        config (object): Configuration object (e.g., argparse Namespace) with run parameters.
        output_dir (str): Directory to save the Markdown report.
        experiment_name (str): Name of the experiment for the report title.

    Returns:
        str: Path to the generated Markdown report.
    """
    report_path = os.path.join(output_dir, f"{experiment_name.replace(' ', '_')}_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Experiment Report: {experiment_name}\n")
        f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

        f.write("## 1. Experiment Configuration\n")
        f.write("```\n")
        for arg, value in sorted(vars(config).items()):
            f.write(f"{arg}: {value}\n")
        f.write("```\n\n")

        # Section for Noisy Tenant Information
        noisy_tenant_info = experiment_results.get('noisy_tenant_info', {})
        if noisy_tenant_info:
            f.write("## 2. Noisy Tenant Identification\n")
            detected_tenant = noisy_tenant_info.get('detected_noisy_tenant', 'Not detected')
            method = noisy_tenant_info.get('detection_method', 'N/A')
            f.write(f"- **Detected Noisy Tenant:** {detected_tenant}\n")
            f.write(f"- **Detection Method:** {method}\n")
            if 'reasoning' in noisy_tenant_info:
                f.write(f"- **Reasoning:** {noisy_tenant_info['reasoning']}\n")
            if 'plot_path' in noisy_tenant_info and noisy_tenant_info['plot_path']:
                relative_plot_path = os.path.relpath(noisy_tenant_info['plot_path'], os.path.dirname(report_path))
                f.write(f"- **Supporting Plot:** ![Noisy Tenant Plot]({relative_plot_path.replace(os.sep, '/')})\n")
            f.write("\n")
        
        current_section = 3

        # Section for Basic Plots and Tables
        f.write(f"## {current_section}. Metrics Overview\n")
        
        main_output_dir = os.path.abspath(os.path.join(output_dir, ".."))
        metric_plots_dir = os.path.join(main_output_dir, "plots") 

        if os.path.exists(metric_plots_dir):
            f.write(f"### {current_section}.1 Metric Behavior by Phase\n")
            plot_files_found = False
            for plot_file in sorted(os.listdir(metric_plots_dir)):
                if plot_file.startswith("metric_by_phase") and plot_file.endswith(".png"):
                    absolute_plot_path = os.path.join(metric_plots_dir, plot_file)
                    relative_plot_path = os.path.relpath(absolute_plot_path, os.path.dirname(report_path))
                    f.write(f"![{plot_file.replace('_', ' ').title()}]({relative_plot_path.replace(os.sep, '/')})\n\n")
                    plot_files_found = True
            if not plot_files_found:
                f.write("*No 'metric by phase' plots found.*\n\n")
            f.write("\n")
        current_section += 1

        # Phase Comparison Tables
        phase_comp_tables = experiment_results.get('phase_comparison_tables', {})
        if phase_comp_tables:
            f.write(f"## {current_section}. Phase Comparison Analysis\n")
            tables_found = False
            for metric, table_path in phase_comp_tables.items():
                if table_path and os.path.exists(table_path):
                    f.write(f"### {current_section}.1 {metric.replace('_', ' ').title()}\n")
                    try:
                        df = pd.read_csv(table_path)
                        f.write(convert_df_to_markdown(df, table_format="pipe"))
                        f.write("\n\n")
                        tables_found = True
                    except Exception as e:
                        f.write(f"*Error loading table {metric}: {e}*\n\n")
            if not tables_found:
                f.write("*No phase comparison tables found or processed.*\n\n")
            current_section += 1
            
        # Impact Summary Tables
        impact_tables = experiment_results.get('impact_summary_tables', {})
        if impact_tables:
            f.write(f"## {current_section}. Tenant Impact Summary\n")
            tables_found = False
            for metric, table_path in impact_tables.items():
                if table_path and os.path.exists(table_path):
                    f.write(f"### {current_section}.1 {metric.replace('_', ' ').title()}\n")
                    try:
                        df = pd.read_csv(table_path)
                        f.write(convert_df_to_markdown(df, table_format="pipe"))
                        f.write("\n\n")
                        tables_found = True
                    except Exception as e:
                        f.write(f"*Error loading table {metric}: {e}*\n\n")
            if not tables_found:
                f.write("*No tenant impact summary tables found or processed.*\n\n")
            current_section += 1

        # Advanced Analysis Section
        if config.advanced or config.inter_tenant_causality:
            f.write(f"## {current_section}. Advanced Analysis\n")
            advanced_content_found = False
            
            # Correlation Matrix Plots
            corr_plots = experiment_results.get('correlation_matrix_plots', {})
            if corr_plots:
                f.write(f"### {current_section}.1 Correlation Matrices\n")
                plots_found = False
                for metric_or_phase, plot_path in corr_plots.items():
                    if plot_path and os.path.exists(plot_path):
                        relative_plot_path = os.path.relpath(plot_path, os.path.dirname(report_path))
                        f.write(f"**{metric_or_phase.replace('_', ' ').title()}**\n")
                        f.write(f"![Correlation Matrix for {metric_or_phase.replace('_', ' ')}]({relative_plot_path.replace(os.sep, '/')})\n\n")
                        plots_found = True
                        advanced_content_found = True
                if not plots_found:
                    f.write("*No correlation matrix plots found.*\n\n")
                f.write("\n")

            # Entropy Results
            entropy_results = experiment_results.get('entropy_results', {})
            if entropy_results:
                f.write(f"### {current_section}.2 Entropy Analysis\n")
                entropy_content_added = False
                if 'plot_paths' in entropy_results:
                    for plot_type, plot_path in entropy_results['plot_paths'].items():
                        if plot_path and os.path.exists(plot_path):
                            relative_plot_path = os.path.relpath(plot_path, os.path.dirname(report_path))
                            f.write(f"**Entropy {plot_type.title()}**\n")
                            f.write(f"![Entropy {plot_type}]({relative_plot_path.replace(os.sep, '/')})\n\n")
                            entropy_content_added = True
                            advanced_content_found = True
                if 'table_path' in entropy_results and entropy_results['table_path'] and os.path.exists(entropy_results['table_path']):
                    try:
                        df_entropy = pd.read_csv(entropy_results['table_path'])
                        f.write("**Top Entropy Pairs Table**\n")
                        f.write(convert_df_to_markdown(df_entropy, table_format="pipe"))
                        f.write("\n\n")
                        entropy_content_added = True
                        advanced_content_found = True
                    except Exception as e:
                        f.write(f"*Error loading entropy table: {e}*\n\n")
                if not entropy_content_added:
                    f.write("*No entropy analysis results found.*\n\n")
                f.write("\n")

            # Causality Results
            causality_results = experiment_results.get('causality_results', {})
            if causality_results:
                f.write(f"### {current_section}.3 Inter-Tenant Causality Analysis\n")
                causality_content_added = False
                causality_plots_base_dir = os.path.join(main_output_dir, "causality", "plots")
                if os.path.exists(causality_plots_base_dir):
                    f.write("**Causal Graphs**\n")
                    found_causality_plots = False
                    for metric_dir in sorted(os.listdir(causality_plots_base_dir)):
                        metric_path = os.path.join(causality_plots_base_dir, metric_dir)
                        if os.path.isdir(metric_path):
                            for phase_dir in sorted(os.listdir(metric_path)):
                                phase_path = os.path.join(metric_path, phase_dir)
                                if os.path.isdir(phase_path):
                                    for plot_file in sorted(os.listdir(phase_path)):
                                        if plot_file.endswith(".png"):
                                            absolute_plot_path = os.path.join(phase_path, plot_file)
                                            relative_plot_path = os.path.relpath(absolute_plot_path, os.path.dirname(report_path))
                                            title_parts = plot_file.split('_')
                                            plot_title = f"{title_parts[1].replace('-', ' ').title()} ({title_parts[2].replace('-', ' ')}) Causal Graph" if len(title_parts) > 3 else plot_file
                                            f.write(f"**{plot_title}**\n")
                                            f.write(f"![{plot_title}]({relative_plot_path.replace(os.sep, '/')})\n\n")
                                            found_causality_plots = True
                                            causality_content_added = True
                                            advanced_content_found = True
                    if not found_causality_plots:
                         f.write("*No causal graphs found.*\n\n")
                
                causality_tables_base_dir = os.path.join(main_output_dir, "causality", "tables")
                if os.path.exists(causality_tables_base_dir):
                    f.write("**Causality Summary Tables**\n")
                    found_causality_tables = False
                    for table_file in sorted(os.listdir(causality_tables_base_dir)):
                        if table_file.endswith(".csv"):
                            absolute_table_path = os.path.join(causality_tables_base_dir, table_file)
                            try:
                                df_causality = pd.read_csv(absolute_table_path)
                                table_title = table_file.replace(".csv", "").replace("_", " ").title()
                                f.write(f"**{table_title}**\n")
                                f.write(convert_df_to_markdown(df_causality, table_format="pipe"))
                                f.write("\n\n")
                                found_causality_tables = True
                                causality_content_added = True
                                advanced_content_found = True
                            except Exception as e:
                                f.write(f"*Error loading causality table {table_file}: {e}*\n\n")
                    if not found_causality_tables:
                        f.write("*No causality summary tables found.*\n\n")

                if not causality_content_added:
                    f.write("*No inter-tenant causality analysis results found or processed.*\n\n")
                f.write("\n")
            
            if not advanced_content_found:
                 f.write("*No advanced analysis results (correlation, entropy, causality) were generated or found.*\n\n")
            current_section +=1

        # Intra-Round Comparison
        rounds_comp = experiment_results.get('rounds_comparison_outputs')
        if config.compare_rounds_intra and rounds_comp:
            f.write(f"## {current_section}. Intra-Experiment Round Comparison\n")
            content_found = False
            for key, data in rounds_comp.items():
                metric_phase = key.replace("_", " ").title()
                f.write(f"### {current_section}.1 {metric_phase}\n")
                item_content_found = False
                if data.get("plot_path") and os.path.exists(data["plot_path"]):
                    relative_plot_path = os.path.relpath(data["plot_path"], os.path.dirname(report_path))
                    f.write(f"![Round Comparison Plot for {metric_phase}]({relative_plot_path.replace(os.sep, '/')})\n\n")
                    item_content_found = True
                if data.get("csv_path") and os.path.exists(data["csv_path"]):
                    try:
                        df_rounds = pd.read_csv(data["csv_path"])
                        f.write("**Summary Statistics per Round**\n")
                        f.write(convert_df_to_markdown(df_rounds, table_format="pipe"))
                        f.write("\n")
                        item_content_found = True
                    except Exception as e:
                        f.write(f"*Error loading round comparison table for {metric_phase}: {e}*\n\n")
                if data.get("anova_f_stat") is not None:
                    f.write(f"- ANOVA F-statistic: {data['anova_f_stat']:.4f}\n")
                    f.write(f"- ANOVA p-value: {data['anova_p_value']:.4f}\n")
                    item_content_found = True
                if not item_content_found:
                    f.write("*No data (plot, table, or ANOVA results) found for this metric/phase.*\n")
                f.write("\n")
                content_found = True
            if not content_found:
                f.write("*No intra-experiment round comparison results found or processed.*\n\n")
            current_section += 1

        # Tenant Comparison (Direct)
        tenant_comp = experiment_results.get('tenant_comparison_outputs')
        if config.compare_tenants_directly and tenant_comp:
            f.write(f"## {current_section}. Direct Tenant Comparison\n")
            content_found = False
            for key, data in tenant_comp.items():
                metric_phase_title = key.replace("_", " ").title()
                f.write(f"### {current_section}.1 {metric_phase_title}\n")
                item_content_found = False
                if data.get("plot_path") and os.path.exists(data["plot_path"]):
                    relative_plot_path = os.path.relpath(data["plot_path"], os.path.dirname(report_path))
                    f.write(f"![Tenant Comparison Plot for {metric_phase_title}]({relative_plot_path.replace(os.sep, '/')})\n\n")
                    item_content_found = True
                if data.get("table_path") and os.path.exists(data["table_path"]):
                    try:
                        df_tenant_comp = pd.read_csv(data["table_path"])
                        f.write("**Statistical Comparison between Tenants**\n")
                        f.write(convert_df_to_markdown(df_tenant_comp, table_format="pipe"))
                        f.write("\n")
                        item_content_found = True
                    except Exception as e:
                        f.write(f"*Error loading tenant comparison table for {metric_phase_title}: {e}*\n\n")
                if not item_content_found:
                    f.write("*No data (plot or table) found for this metric/phase.*\n")
                f.write("\n")
                content_found = True
            if not content_found:
                f.write("*No direct tenant comparison results found or processed.*\n\n")
            current_section += 1
            
        # Application Metrics Analysis
        app_metrics_results = experiment_results.get('app_metrics_analysis_results')
        if config.app_metrics_analysis and app_metrics_results:
            f.write(f"## {current_section}. Application Metrics Analysis\n")
            app_content_found = False
            if 'latency_impact' in app_metrics_results and app_metrics_results['latency_impact']:
                f.write(f"### {current_section}.1 Latency Impact\n")
                latency_df = pd.DataFrame.from_dict(app_metrics_results['latency_impact'], orient='index')
                latency_df.index.name = 'Tenant'
                latency_df = latency_df.reset_index()
                latency_df.columns = [col.replace('_', ' ').title() for col in latency_df.columns]
                f.write(convert_df_to_markdown(latency_df, table_format="pipe"))
                f.write("\n\n")
                app_content_found = True
            if 'error_correlations' in app_metrics_results and app_metrics_results['error_correlations']:
                f.write(f"### {current_section}.2 Error Rate Correlations with Noisy Tenant CPU\n")
                error_df = pd.DataFrame(list(app_metrics_results['error_correlations'].items()), columns=['Tenant', 'Correlation with Noisy Tenant CPU'])
                f.write(convert_df_to_markdown(error_df, table_format="pipe"))
                f.write("\n\n")
                app_content_found = True
            if 'slo_violations' in app_metrics_results and app_metrics_results['slo_violations']:
                f.write(f"### {current_section}.3 SLO Violations\n")
                slo_content_for_report = False
                for tenant, violations in app_metrics_results['slo_violations'].items():
                    f.write(f"**Tenant: {tenant}**\n")
                    violation_data = []
                    for metric, stats in violations.items():
                        violation_data.append({
                            'Metric': metric, 
                            'Violation Increase (%)': f"{stats.get('violation_increase', 0) * 100:.2f}",
                            'Affected Duration (s)': f"{stats.get('affected_duration_seconds', 'N/A')}"
                        })
                    if violation_data:
                        df_violations = pd.DataFrame(violation_data)
                        f.write(convert_df_to_markdown(df_violations, table_format="pipe"))
                        f.write("\n")
                        slo_content_for_report = True
                    else:
                        f.write("*No SLO violations recorded or data available for this tenant.*\n")
                    f.write("\n")
                if slo_content_for_report:
                    app_content_found = True
            
            if not app_content_found:
                f.write("*No application metrics analysis results found or processed.*\n\n")
            current_section += 1

        # Experiment Comparison (Multiple Experiments)
        comp_exp_results = experiment_results.get('comparison_experiment_results')
        if config.compare_experiments and comp_exp_results:
            f.write(f"## {current_section}. Comparison with Other Experiments\n")
            comp_content_found = False
            if 'summary_plots' in comp_exp_results and comp_exp_results['summary_plots']:
                plots_found_comp = False
                for plot_name, plot_path in comp_exp_results['summary_plots'].items():
                    if plot_path and os.path.exists(plot_path):
                        relative_plot_path = os.path.relpath(plot_path, os.path.dirname(report_path))
                        f.write(f"### {plot_name.replace('_', ' ').title()}\n")
                        f.write(f"![{plot_name.replace('_', ' ')}]({relative_plot_path.replace(os.sep, '/')})\n\n")
                        plots_found_comp = True
                if plots_found_comp: comp_content_found = True
            
            if 'summary_tables' in comp_exp_results and comp_exp_results['summary_tables']:
                tables_found_comp = False
                for table_name, table_path in comp_exp_results['summary_tables'].items():
                    if table_path and os.path.exists(table_path):
                        try:
                            df_comp = pd.read_csv(table_path)
                            f.write(f"**{table_name.replace('_', ' ').title()}**\n")
                            f.write(convert_df_to_markdown(df_comp, table_format="pipe"))
                            f.write("\n\n")
                            tables_found_comp = True
                        except Exception as e:
                            f.write(f"*Error loading comparison table {table_name}: {e}*\n\n")
                if tables_found_comp: comp_content_found = True
            
            if not comp_content_found:
                f.write("*No comparison results with other experiments found or processed.*\n\n")
            current_section += 1

        # Technology Comparison
        tech_comp_results = experiment_results.get('technology_comparison_results')
        if config.compare_technologies and tech_comp_results:
            f.write(f"## {current_section}. Technology Comparison\n")
            tech_content_found = False
            if 'plots' in tech_comp_results and tech_comp_results['plots']:
                plots_found_tech = False
                for plot_name, plot_path in tech_comp_results['plots'].items():
                    if plot_path and os.path.exists(plot_path):
                        relative_plot_path = os.path.relpath(plot_path, os.path.dirname(report_path))
                        f.write(f"### {plot_name.replace('_', ' ').title()}\n")
                        f.write(f"![{plot_name.replace('_', ' ')}]({relative_plot_path.replace(os.sep, '/')})\n\n")
                        plots_found_tech = True
                if plots_found_tech: tech_content_found = True
            if 'efficiency_metrics' in tech_comp_results and tech_comp_results['efficiency_metrics']:
                all_phases_summary_path = tech_comp_results['efficiency_metrics'].get('all_phases_csv_path')
                if all_phases_summary_path and os.path.exists(all_phases_summary_path):
                    try:
                        df_tech_eff = pd.read_csv(all_phases_summary_path)
                        f.write("**Overall Efficiency Comparison**\n")
                        f.write(convert_df_to_markdown(df_tech_eff, table_format="pipe"))
                        f.write("\n\n")
                        tech_content_found = True
                    except Exception as e:
                        f.write(f"*Error loading technology efficiency table: {e}*\n\n")
            
            if not tech_content_found:
                f.write("*No technology comparison results found or processed.*\n\n")
            current_section += 1
            
        f.write("## End of Report\n")

    print(f"Markdown report generated: {report_path}")
    return report_path

def generate_html_report(markdown_report_path, output_dir, experiment_name="Experiment"):
    """
    Generates an HTML report from a Markdown file.
    (This is a basic version using a Markdown library, could be enhanced)

    Args:
        markdown_report_path (str): Path to the Markdown report file.
        output_dir (str): Directory to save the HTML report.
        experiment_name (str): Name for the HTML file.

    Returns:
        str: Path to the generated HTML report.
    """
    try:
        import markdown
    except ImportError:
        print("Markdown library not found. Skipping HTML report generation. Install with: pip install markdown")
        return None

    html_report_path = os.path.join(output_dir, f"{experiment_name.replace(' ', '_')}_report.html")
    
    with open(markdown_report_path, 'r', encoding='utf-8') as f_md:
        md_content = f_md.read()

    html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code', 'md_in_html'])

    html_output = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{experiment_name} Report</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol"; margin: 20px; line-height: 1.6; color: #333; background-color: #fdfdfd; }}
            .container {{ max-width: 900px; margin: 0 auto; background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.05); }}
            h1, h2, h3, h4, h5, h6 {{ color: #2c3e50; margin-top: 1.5em; margin-bottom: 0.5em; }}
            h1 {{ border-bottom: 2px solid #ecf0f1; padding-bottom: 0.3em; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; border: 1px solid #ddd; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; vertical-align: top; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            img {{ max-width: 100%; height: auto; display: block; margin-top: 10px; margin-bottom: 20px; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            pre {{ background-color: #f5f5f5; padding: 15px; border: 1px solid #eee; border-radius: 5px; overflow-x: auto; font-size: 0.9em; }}
            code {{ font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace; background-color: #f0f0f0; padding: 0.2em 0.4em; border-radius: 3px; font-size: 0.9em;}}
            pre > code {{ background-color: transparent; padding: 0; border-radius: 0; font-size: inherit;}}
            a {{ color: #3498db; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
            ul, ol {{ padding-left: 20px; }}
            li {{ margin-bottom: 0.5em; }}
        </style>
    </head>
    <body>
        <div class="container">
            {html_content}
        </div>
    </body>
    </html>
    """

    with open(html_report_path, 'w', encoding='utf-8') as f_html:
        f_html.write(html_output)
        
    print(f"HTML report generated: {html_report_path}")
    return html_report_path

def generate_all_reports(experiment_results, config, output_dir_reports, experiment_name="Experiment"):
    """
    Generates all supported report formats (Markdown, HTML).

    Args:
        experiment_results (dict): Results from the pipeline.
        config (object): Configuration object.
        output_dir_reports (str): Base directory for reports.
        experiment_name (str): Name of the experiment.
    """
    os.makedirs(output_dir_reports, exist_ok=True)

    md_path = generate_markdown_report(experiment_results, config, output_dir_reports, experiment_name)
    
    if md_path:
        html_path = generate_html_report(md_path, output_dir_reports, experiment_name)

    print("\nReport generation finished.")

if __name__ == '__main__':
    print("Running report generator example...")
    
    class DummyConfig:
        def __init__(self):
            self.data_dir = "/path/to/data"
            self.output_dir = "./dummy_output_for_reports"
            self.tenants = ["tenant-a", "tenant-b"]
            self.metrics = ["cpu_usage", "memory_usage"]
            self.advanced = True
            self.inter_tenant_causality = True
            self.compare_rounds_intra = True
            self.compare_tenants_directly = True
            self.app_metrics_analysis = True
            self.compare_experiments = True
            self.compare_technologies = True
            self.experiment_name = "Dummy_Experiment_Test"

    dummy_config = DummyConfig()
    reports_output_dir = os.path.join(dummy_config.output_dir, "reports")
    os.makedirs(reports_output_dir, exist_ok=True)

    base_pipeline_output_dir = dummy_config.output_dir
    os.makedirs(os.path.join(base_pipeline_output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(base_pipeline_output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(base_pipeline_output_dir, "causality", "plots", "cpu_usage", "1_-_Baseline"), exist_ok=True)
    os.makedirs(os.path.join(base_pipeline_output_dir, "causality", "tables"), exist_ok=True)
    os.makedirs(os.path.join(base_pipeline_output_dir, "rounds_comparison_intra", "plots"), exist_ok=True)
    os.makedirs(os.path.join(base_pipeline_output_dir, "rounds_comparison_intra"), exist_ok=True)
    os.makedirs(os.path.join(base_pipeline_output_dir, "tenant_comparison", "plots"), exist_ok=True)
    os.makedirs(os.path.join(base_pipeline_output_dir, "tenant_comparison"), exist_ok=True)
    os.makedirs(os.path.join(base_pipeline_output_dir, "experiment_comparison_outputs", "plots"), exist_ok=True)
    os.makedirs(os.path.join(base_pipeline_output_dir, "experiment_comparison_outputs", "tables"), exist_ok=True)
    os.makedirs(os.path.join(base_pipeline_output_dir, "technology_comparison_outputs", "plots"), exist_ok=True)
    os.makedirs(os.path.join(base_pipeline_output_dir, "technology_comparison_outputs", "tables"), exist_ok=True)

    dummy_experiment_results = {
        'noisy_tenant_info': {
            'detected_noisy_tenant': 'tenant-b',
            'detection_method': 'Max Standard Deviation during Attack Phase',
            'reasoning': 'Tenant B showed the highest CPU usage volatility during the attack phase.',
            'plot_path': os.path.join(base_pipeline_output_dir, "plots", "dummy_noisy_tenant_plot.png")
        },
        'phase_comparison_tables': {
            'cpu_usage': os.path.join(base_pipeline_output_dir, "tables", "cpu_usage_phase_comparison.csv"),
        },
        'impact_summary_tables': {
            'cpu_usage': os.path.join(base_pipeline_output_dir, "tables", "cpu_usage_impact_summary.csv"),
        },
        'correlation_matrix_plots': {
            'cpu_usage_all_tenants_correlation': os.path.join(base_pipeline_output_dir, "plots", "cpu_usage_all_tenants_correlation.png")
        },
        'entropy_results': {
            'plot_paths': {
                'entropy_heatmap': os.path.join(base_pipeline_output_dir, "plots", "entropy_heatmap.png"),
                'entropy_top_pairs_barplot': os.path.join(base_pipeline_output_dir, "plots", "entropy_top_pairs_barplot.png")
            },
            'table_path': os.path.join(base_pipeline_output_dir, "tables", "entropy_top_pairs.csv")
        },
        'causality_results': {},
        'rounds_comparison_outputs': {
            'cpu_usage_2_-_Attack': {
                'csv_path': os.path.join(base_pipeline_output_dir, "rounds_comparison_intra", "cpu_usage_2_-_Attack_round_stats.csv"),
                'plot_path': os.path.join(base_pipeline_output_dir, "rounds_comparison_intra", "plots", "cpu_usage_2_-_Attack_round_comparison_plot.png"),
                'anova_f_stat': 5.67,
                'anova_p_value': 0.034
            }
        },
        'tenant_comparison_outputs': {
            'cpu_usage_2_-_Attack': {
                'table_path': os.path.join(base_pipeline_output_dir, "tenant_comparison", "cpu_usage_2_-_Attack_tenant_comparison_stats.csv"),
                'plot_path': os.path.join(base_pipeline_output_dir, "tenant_comparison", "plots", "cpu_usage_2_-_Attack_tenant_comparison_plot.png")
            }
        },
        'app_metrics_analysis_results': {
            'latency_impact': {'tenant-a': {'average_latency_increase_ms': 50.5, 'significance_p_value': 0.01, 'impact_level': 'High'}},
            'error_correlations': {'tenant-a_vs_noisy_cpu': 0.75},
            'slo_violations': {
                'tenant-a': {
                    'latency_ms_slo': {'violation_increase': 0.2, 'affected_duration_seconds': 120, 'details': 'Exceeded 200ms threshold'}
                }
            }
        },
        'comparison_experiment_results': {
            'summary_plots': {
                'average_cpu_usage_across_experiments': os.path.join(base_pipeline_output_dir, "experiment_comparison_outputs", "plots", "avg_cpu_usage_comparison.png")
            },
            'summary_tables': {
                'key_metrics_summary_across_experiments': os.path.join(base_pipeline_output_dir, "experiment_comparison_outputs", "tables", "key_metrics_summary_comparison.csv")
            }
        },
        'technology_comparison_results': {
            'plots': {
                'cpu_efficiency_comparison_by_technology': os.path.join(base_pipeline_output_dir, "technology_comparison_outputs", "plots", "tech_cpu_efficiency.png")
            },
            'efficiency_metrics': { 
                'all_phases_csv_path': os.path.join(base_pipeline_output_dir, "technology_comparison_outputs", "tables", "tech_efficiency_all_phases.csv")
            }
        }
    }

    open(dummy_experiment_results['noisy_tenant_info']['plot_path'], 'a').close()
    open(dummy_experiment_results['correlation_matrix_plots']['cpu_usage_all_tenants_correlation'], 'a').close()
    open(dummy_experiment_results['entropy_results']['plot_paths']['entropy_heatmap'], 'a').close()
    open(dummy_experiment_results['entropy_results']['plot_paths']['entropy_top_pairs_barplot'], 'a').close()
    
    dummy_causality_plot_path = os.path.join(base_pipeline_output_dir, "causality", "plots", "cpu_usage", "1_-_Baseline", "experiment_cpu_usage_1_-_Baseline_causal_graph.png")
    open(dummy_causality_plot_path, 'a').close()
    
    pd.DataFrame({'Phase': ['Baseline', 'Attack'], 'Mean CPU': [10, 50]}).to_csv(dummy_experiment_results['phase_comparison_tables']['cpu_usage'], index=False)
    pd.DataFrame({'Tenant': ['A','B'], 'CPU Impact (%)': [20.5, 15.8]}).to_csv(dummy_experiment_results['impact_summary_tables']['cpu_usage'], index=False)
    pd.DataFrame({'Pair': ['tenant-a_cpu-tenant-b_cpu'], 'Mutual Information': [1.2]}).to_csv(dummy_experiment_results['entropy_results']['table_path'], index=False)
    
    dummy_causality_table_path = os.path.join(base_pipeline_output_dir, "causality", "tables", "cpu_usage_1_-_Baseline_causality_summary.csv")
    pd.DataFrame({'Source Tenant': ['A'], 'Target Tenant':['B'], 'P-Value': [0.04], 'Lag': [2]}).to_csv(dummy_causality_table_path, index=False)

    pd.DataFrame({'Round': [1,2,3], 'Mean CPU Usage': [10,12,11]}).to_csv(dummy_experiment_results['rounds_comparison_outputs']['cpu_usage_2_-_Attack']['csv_path'], index=False)
    open(dummy_experiment_results['rounds_comparison_outputs']['cpu_usage_2_-_Attack']['plot_path'], 'a').close()

    pd.DataFrame({'Tenant': ['A','B'], 'Mean CPU (Attack)': [50, 55], 'P-Value (vs A)': [None, 0.04]}).to_csv(dummy_experiment_results['tenant_comparison_outputs']['cpu_usage_2_-_Attack']['table_path'], index=False)
    open(dummy_experiment_results['tenant_comparison_outputs']['cpu_usage_2_-_Attack']['plot_path'], 'a').close()
    
    open(dummy_experiment_results['comparison_experiment_results']['summary_plots']['average_cpu_usage_across_experiments'], 'a').close()
    pd.DataFrame({'Experiment ID': ['Exp1','Exp2'], 'Average Noisy Tenant CPU (%)': [60,65]}).to_csv(dummy_experiment_results['comparison_experiment_results']['summary_tables']['key_metrics_summary_across_experiments'], index=False)

    open(dummy_experiment_results['technology_comparison_results']['plots']['cpu_efficiency_comparison_by_technology'], 'a').close()
    pd.DataFrame({'Technology': ['KSM', 'None'], 'CPU Overhead Reduction (%)': [15.2, 0]}).to_csv(dummy_experiment_results['technology_comparison_results']['efficiency_metrics']['all_phases_csv_path'], index=False)

    generate_all_reports(dummy_experiment_results, dummy_config, reports_output_dir, experiment_name=dummy_config.experiment_name)
    
    print(f"\nExample finished. Check the directory: {reports_output_dir}")
else:
    pass
