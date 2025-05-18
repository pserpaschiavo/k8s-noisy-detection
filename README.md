# Kubernetes Noisy Neighbors Analysis Pipeline

This pipeline performs data analysis on Kubernetes "noisy neighbors" experiments, with a focus on detecting and measuring the impact of resource interference between tenants.

## Features

### 1. Data Processing
- Data loading and preprocessing
- Time normalization
- Data aggregation and statistics

### 2. Basic Analysis
- Tenant comparison
- Phase analysis (baseline, stress, recovery)
- Impact calculation

### 3. Advanced Analysis
- Correlation and covariance analysis
- Entropy metrics
- Granger causality testing
- Automatic noisy tenant detection

### 4. Anomaly Detection
- Isolation Forest algorithm
- Local Outlier Factor (LOF)
- Change point detection
- Multivariate pattern change detection

### 5. Experiment Comparison
- Statistical comparison across experiments
- Distribution testing (K-S test)

### 6. Visualization
- Various plots and charts for all analysis types
- Anomaly visualization
- Comparison plots

### 7. Report Generation
- Markdown reports
- LaTeX reports with publication-ready quality
- HTML reports with interactive elements
- Customizable templates

### 9. Application-Level Analysis
- Service-level objective (SLO) violation detection
- Latency impact analysis
- Error rate correlation with resource usage
- Application performance metrics

## Usage

```bash
python -m pipeline.main --data-dir=/path/to/experiment/data --output-dir=output \
  --advanced --anomaly-detection --generate-reports
```

### Command-line Arguments:

```
--data-dir DIR       : Directory with experiment data
--compare-dir DIR... : Additional directories for comparing multiple experiments
--output-dir DIR     : Directory to save results (default: output)
--tenants T...       : Specific tenant(s) to analyze
--noisy-tenant T     : Specific tenant that generates noise (default: tenant-b)
--auto-detect-noisy  : Automatically detect which tenant is the noisy neighbor
--metrics M...       : Specific metric(s) to analyze
--phases P...        : Specific phase(s) to analyze
--rounds R...        : Specific round(s) to analyze
--skip-plots         : Skip plot generation
--skip-tables        : Skip table generation
--show-as-percentage : Show metrics as percentage of total cluster resources
--advanced           : Run advanced analyses (covariance, entropy, causality)
--anomaly-detection  : Run anomaly detection
--compare-experiments: Compare multiple experiments
--generate-reports   : Generate reports in Markdown, LaTeX and HTML
```

### Raw Metrics Analysis

To analyze raw metrics without normalization or unit conversion:

```bash
# Generate raw metrics visualizations as part of the main pipeline
python -m pipeline.main --data-dir=/path/to/experiment/data --raw-metrics --raw-metrics-style=bar

# Or use the standalone raw metrics analyzer for more options
python -m pipeline.raw_metrics.analyzer --output-dir=output/raw_metrics --comparison-type=all --plot-style=line

# Generate all available visualization styles at once
python generate_all_raw_plots.py
```

This is helpful for:
1. Comparing raw vs. normalized values
2. Validating percentage calculations
3. Analyzing metrics in their original scale
4. Understanding data before unit conversion

## Report Templates

The pipeline includes customizable templates for report generation:

- **Markdown**: `/pipeline/templates/markdown_report.md`
- **LaTeX**: `/pipeline/templates/latex_report.tex`
- **HTML**: `/pipeline/templates/html_report.html`

You can customize these templates to fit your specific reporting needs.

## Example

To run a complete analysis with all features:

```bash
python -m pipeline.main \
  --data-dir=/path/to/experiment \
  --output-dir=results \
  --advanced \
  --anomaly-detection \
  --generate-reports
```

To compare multiple experiments:

```bash
python -m pipeline.main \
  --data-dir=/path/to/primary/experiment \
  --compare-dir=/path/to/experiment2 /path/to/experiment3 \
  --output-dir=comparison_results \
  --compare-experiments \
  --generate-reports
```

To run analysis with automatic noisy neighbor detection:

```bash
python -m pipeline.main \
  --data-dir=/path/to/experiment \
  --output-dir=results \
  --auto-detect-noisy \
  --generate-reports
```

To analyze raw metrics (without normalization):

```bash
python -m pipeline.main \
  --data-dir=/path/to/experiment \
  --output-dir=results \
  --raw-metrics \
  --raw-metrics-style=line
```

You can also use the standalone raw metrics analyzer:

```bash
python -m pipeline.raw_metrics.analyzer \
  --data-dir=/path/to/experiment \
  --output-dir=output/raw_analysis \
  --comparison-type=all \
  --plot-style=box \
  --show-values
```

Or generate all styles of visualizations at once:

```bash
python -m pipeline.raw_metrics.generate_all_plots
```

For advanced testing and examples, use the testing snippets:

```bash
python testing_snippets.py
```

This interactive menu provides access to various code snippets that demonstrate different analysis techniques:
python -m pipeline.main \
  --data-dir=/path/to/experiment \
  --output-dir=results \
  --auto-detect-noisy \
  --advanced \
  --anomaly-detection
```

To specify a custom noisy tenant:

```bash
python -m pipeline.main \
  --data-dir=/path/to/experiment \
  --output-dir=results \
  --noisy-tenant=tenant-c
```

To show metrics as percentage of cluster resources:

```bash
python -m pipeline.main \
  --data-dir=/path/to/experiment \
  --output-dir=results \
  --show-as-percentage \
  --advanced \
  --anomaly-detection
```

## Testing

Run the test suite to validate functionality:

```bash
python -m pipeline.test_report_templates
```

This will test the entire pipeline, including template rendering and report generation.

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy
- jinja2
- tabulate

## Author

Created as part of the Kubernetes Noisy Neighbors Experiment project.
