# Report Generator for Kubernetes Noisy Neighbors Analysis

This directory contains templates for generating reports about the experiment results in three formats:

- Markdown
- LaTeX
- HTML

## Using the Report Generator

The report generator is implemented in the `pipeline/visualization/report_generator.py` module. It can produce comprehensive reports from experiment data in different formats:

```python
from pipeline.visualization.report_generator import ReportGenerator

# Initialize the report generator
report_gen = ReportGenerator(
    output_dir='path/to/output/directory',  # Where to save the reports
    template_dir='path/to/template/directory'  # Optional, defaults to pipeline/templates
)

# Generate a Markdown report
markdown_path = report_gen.generate_markdown_report(
    metrics_data=metrics_data,              # Dict of pandas DataFrames with metrics
    analysis_results=analysis_results,      # Dict with basic analysis results
    advanced_results=advanced_results,      # Dict with advanced analysis results (optional)
    anomaly_results=anomaly_results,        # Dict with anomaly detection results (optional)
    comparison_results=comparison_results,  # Dict with experiment comparison results (optional)
    include_figures=True,                   # Whether to include figures in the report
    figure_paths=figure_paths               # Dict with paths to figures (optional)
)

# Generate a LaTeX report
latex_path = report_gen.generate_latex_report(
    # Same parameters as generate_markdown_report
)

# Generate an HTML report
html_path = report_gen.generate_html_report(
    # Same parameters as generate_markdown_report
)
```

## Template Structure

The report templates use Jinja2 for templating. The following context variables are available:

- `title`: The title of the report
- `date`: The date when the report is generated
- `experiment`: Information about the experiment (metrics, tenants, phases, rounds, etc.)
- `analysis_results`: Results of basic analyses
- `advanced_results`: Results of advanced analyses
- `anomaly_results`: Results of anomaly detection
- `comparison_results`: Results of experiment comparison
- `include_figures`: Whether to include figures in the report
- `figure_paths`: Paths to figures

## Custom Templates

You can create custom templates by:

1. Creating your own template files in a directory
2. Initializing the ReportGenerator with your template directory
3. Generating reports as normal

## Testing Templates

You can test templates using the provided test script:

```bash
python -u test_templates.py
```

This will generate test reports in Markdown, LaTeX, and HTML formats using sample data.

## Example Integration with Main Pipeline

Here's an example of how to integrate report generation into the main pipeline:

```python
# In main.py or similar orchestration script
def generate_reports(experiment_dir, output_dir, metrics_data, analysis_results, 
                     advanced_results, anomaly_results, comparison_results=None):
    """Generate comprehensive reports of the experiment."""
    
    # Prepare figure paths dictionary (if needed)
    figure_paths = {
        "metrics": {...},
        "anomalies": {...},
        "comparisons": {...}
    }
    
    # Initialize report generator
    report_gen = ReportGenerator(output_dir=os.path.join(output_dir, 'reports'))
    
    # Generate reports in each format
    markdown_path = report_gen.generate_markdown_report(
        metrics_data=metrics_data,
        analysis_results=analysis_results,
        advanced_results=advanced_results,
        anomaly_results=anomaly_results,
        comparison_results=comparison_results,
        include_figures=True,
        figure_paths=figure_paths
    )
    
    latex_path = report_gen.generate_latex_report(...)
    html_path = report_gen.generate_html_report(...)
    
    print(f"Reports generated: {markdown_path}, {latex_path}, {html_path}")
    
    return {
        'markdown': markdown_path,
        'latex': latex_path,
        'html': html_path
    }
```
