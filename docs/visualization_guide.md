# Visualization Functions Guide for k8s-noisy-detection

This guide explains the visualization functions available for analyzing PCA, ICA and comparative results in the k8s-noisy-detection project.

## Table of Contents
- [PCA Visualizations](#pca-visualizations)
- [ICA Visualizations](#ica-visualizations)
- [PCA vs ICA Comparison Visualizations](#pca-vs-ica-comparison-visualizations)
- [Using Visualizations in Your Analysis](#using-visualizations-in-your-analysis)

## PCA Visualizations

### 1. Scree Plot (Explained Variance Plot)
**Function:** `plot_pca_explained_variance()`

Visualizes the amount of variance explained by each principal component. This helps you decide how many components to retain for your analysis.

```python
plot_pca_explained_variance(
    explained_variance_ratio,   # Array of explained variance ratios 
    cumulative_variance=None,   # Optional: cumulative variance array
    title="PCA Explained Variance",
    output_dir=None,            # Directory to save the plot
    filename=None,              # Filename for saving
    metric_name=None,           # Name of the metric being analyzed
    round_name=None,            # Experiment round name
    phase_name=None,            # Experiment phase name
    threshold_line=0.8          # Horizontal line showing variance threshold
)
```

### 2. PCA Biplot
**Function:** `plot_pca_biplot()`

Creates a combined plot showing both:
- Sample projections on principal components
- Loadings vectors showing how each original variable contributes to the components

```python
plot_pca_biplot(
    pca_results,               # DataFrame with PC scores
    pca_components,            # DataFrame with component loadings
    x_component=1,             # Component for x-axis (1-based)
    y_component=2,             # Component for y-axis (1-based)
    scale_arrows=1.0,          # Scaling factor for loading vectors
    sample_groups=None,        # Optional: grouping for samples (e.g., tenants)
    palette=None,              # Optional: color mapping
    title="PCA Biplot",
    output_dir=None,
    filename=None,
    metric_name=None,
    round_name=None,
    phase_name=None,
    max_features_to_show=15,    # Limits number of variable vectors
    arrow_alpha=0.5             # Transparency of arrows
)
```

### 3. PCA Loadings Heatmap
**Function:** `plot_pca_loadings_heatmap()`

Creates a heatmap showing the contribution of each original variable to each principal component.

```python
plot_pca_loadings_heatmap(
    pca_components,           # DataFrame with component loadings 
    title="PCA Loadings",
    output_dir=None,
    filename=None,
    metric_name=None,
    round_name=None,
    phase_name=None,
    cmap='coolwarm',          # Colormap for heatmap
    n_components=None         # Optional: limit to N components
)
```

## ICA Visualizations

### 1. ICA Components Heatmap
**Function:** `plot_ica_components_heatmap()`

Creates a heatmap showing the contribution of each original variable to each independent component.

```python
plot_ica_components_heatmap(
    ica_components,            # DataFrame with ICA components/unmixing matrix
    title="ICA Components Heatmap",
    output_dir=None,
    filename=None,
    metric_name=None,
    round_name=None,
    phase_name=None,
    cmap='coolwarm',           # Colormap for heatmap
    n_components=None          # Optional: limit to N components
)
```

### 2. ICA Time Series
**Function:** `plot_ica_time_series()`

Creates a plot showing the values of independent components over time.

```python
plot_ica_time_series(
    ica_results,              # DataFrame with independent components
    title="ICA Time Series",
    output_dir=None,
    filename=None,
    metric_name=None,
    round_name=None,
    phase_name=None,
    max_components=4          # Maximum components to show
)
```

### 3. ICA Scatter Plot
**Function:** `plot_ica_scatter()`

Creates a scatter plot of two independent components.

```python
plot_ica_scatter(
    ica_results,              # DataFrame with independent components
    x_component=1,            # Component for x-axis (1-based)
    y_component=2,            # Component for y-axis (1-based)
    title="ICA Scatter Plot",
    output_dir=None,
    filename=None,
    metric_name=None,
    round_name=None,
    phase_name=None,
    sample_groups=None,       # Optional: grouping for samples (e.g., tenants)
    palette=None              # Optional: color mapping
)
```

## PCA vs ICA Comparison Visualizations

### 1. Feature Importance Comparison
**Function:** `plot_pca_vs_ica_comparison()`

Creates a bar chart comparing the importance of features in PCA and ICA.

```python
plot_pca_vs_ica_comparison(
    pca_components,           # DataFrame with PCA loadings
    ica_components,           # DataFrame with ICA components
    feature_subset=None,      # Optional: specific features to compare
    n_components=2,           # Number of components to include
    title="PCA vs ICA Feature Importance Comparison",
    output_dir=None,
    filename=None,
    metric_name=None,
    round_name=None,
    phase_name=None
)
```

### 2. Overlay Scatter Plot
**Function:** `plot_pca_vs_ica_overlay_scatter()`

Creates a scatter plot overlaying PCA and ICA results for direct comparison.

```python
plot_pca_vs_ica_overlay_scatter(
    pca_results,             # DataFrame with principal components  
    ica_results,             # DataFrame with independent components
    x_component=1,           # Component for x-axis (1-based)
    y_component=2,           # Component for y-axis (1-based)
    title="PCA vs ICA Overlay Scatter Plot",
    output_dir=None,
    filename=None,
    metric_name=None,
    round_name=None,
    phase_name=None,
    sample_groups=None       # Optional: grouping for samples (e.g., tenants)
)
```

## Using Visualizations in Your Analysis

### Basic Usage
To generate visualizations, you'll need to:

1. Perform PCA and/or ICA analysis to get components and results dataframes
2. Call the appropriate visualization functions with your data
3. Close figures after use to prevent memory issues

Example:

```python
from refactor.visualization.new_plots import (
    plot_pca_explained_variance, plot_pca_biplot, plot_pca_loadings_heatmap,
    plot_ica_components_heatmap, plot_ica_time_series, plot_ica_scatter,
    plot_pca_vs_ica_comparison, plot_pca_vs_ica_overlay_scatter
)

# After running PCA:
pca_results_df, pca_components_df, pca_explained_variance = perform_pca(data)

# Generate PCA visualizations
plot_pca_explained_variance(
    pca_explained_variance,
    title="PCA Explained Variance",
    output_dir="output/plots",
    filename="my_metric_pca_explained_variance.png"
)
plt.close()  # Close the figure after saving

# After running ICA:
ica_results_df, ica_components_df = perform_ica(data)

# Compare PCA and ICA
plot_pca_vs_ica_overlay_scatter(
    pca_results_df,
    ica_results_df,
    title="PCA vs ICA Comparison",
    output_dir="output/plots",
    filename="my_metric_pca_vs_ica_comparison.png"
)
plt.close()  # Close the figure after saving
```

### Best Practices

1. **Always close figures** after they're saved to avoid memory issues, especially when processing many metrics or phases
2. **Use appropriate titles and filenames** that include metric name, round, and phase information
3. **Pass tenant information** via the `sample_groups` parameter when available for grouped visualizations
4. **Set reasonable limits** for features and components when dealing with high-dimensional data
5. **Handle exceptions** when generating visualizations to prevent pipeline failures

### Command Line Usage

The visualizations are automatically generated when running the analysis with the appropriate flags:

```bash
python -m refactor.new_main --data-dir your-data-dir --output-dir output-dir --run-pca --run-ica --compare-pca-ica
```

Available flags:
- `--run-pca`: Generates PCA visualizations
- `--run-ica`: Generates ICA visualizations
- `--compare-pca-ica`: Generates comparative visualizations
