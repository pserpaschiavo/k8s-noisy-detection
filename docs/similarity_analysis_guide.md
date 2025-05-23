# Similarity Analysis Guide for k8s-noisy-detection

This guide explains the similarity analysis techniques available in the k8s-noisy-detection project for detecting similarities and dependencies between time series.

## Table of Contents
- [Distance Correlation (dCor)](#distance-correlation)
- [Cosine Similarity](#cosine-similarity)
- [Dynamic Time Warping (DTW)](#dynamic-time-warping)
- [Using Similarity Analysis in Your Pipeline](#using-similarity-analysis-in-your-pipeline)

## Distance Correlation

Distance Correlation (dCor) is a measure of dependence between random variables. Unlike Pearson correlation, distance correlation can detect nonlinear relationships and is zero if and only if the variables are statistically independent.

### Implementation

```python
from refactor.analysis_modules.similarity import calculate_pairwise_distance_correlation, plot_distance_correlation_heatmap

# Calculate pairwise distance correlation matrix
dcor_matrix = calculate_pairwise_distance_correlation(
    data_df,           # DataFrame containing time series data
    time_col,          # Name of the column representing time
    metric_col,        # Name of the column representing the metric to analyze
    group_col,         # Name of the column representing the groups (e.g., 'tenant_id')
    min_observations   # Minimum observations required for a valid series
)

# Visualize the distance correlation matrix
plot_distance_correlation_heatmap(
    dcor_matrix,       # DataFrame containing the correlation values
    title,             # Title for the plot
    output_dir,        # Directory to save the plot
    filename,          # Base filename for the plot (without extension)
    cmap="viridis",    # Colormap for the heatmap
    fmt=".2f",         # Format for cell annotations
    annot=True,        # Whether to show annotations
    tables_dir         # Optional: directory to save the CSV table
)
```

### Interpretation

- Values range from 0 (no dependence) to 1 (strong dependence)
- High distance correlation indicates statistical dependence between time series
- Particularly useful for detecting nonlinear relationships

## Cosine Similarity

Cosine Similarity measures the cosine of the angle between two vectors, providing a similarity metric that is independent of magnitude and focused on orientation in the feature space.

### Implementation

```python
from refactor.analysis_modules.similarity import calculate_pairwise_cosine_similarity, plot_cosine_similarity_heatmap

# Calculate pairwise cosine similarity matrix
cosine_matrix = calculate_pairwise_cosine_similarity(
    data_df,           # DataFrame containing time series data
    time_col,          # Name of the column representing time
    metric_col,        # Name of the column representing the metric to analyze
    group_col,         # Name of the column representing the groups (e.g., 'tenant')
    min_observations   # Minimum observations required for a valid series
)

# Visualize the cosine similarity matrix
plot_cosine_similarity_heatmap(
    cosine_matrix,     # DataFrame containing the similarity values
    title,             # Title for the plot
    output_dir,        # Directory to save the plot
    filename,          # Base filename for the plot (without extension)
    cmap="viridis",    # Colormap for the heatmap
    fmt=".2f",         # Format for cell annotations
    annot=True,        # Whether to show annotations
    tables_dir         # Optional: directory to save the CSV table
)
```

### Interpretation

- Values range from -1 (exactly opposite) to 1 (exactly the same)
- Values close to 1 indicate high similarity in direction
- Values close to 0 indicate orthogonality (no similarity)
- Values close to -1 indicate opposite patterns

## Dynamic Time Warping

Dynamic Time Warping (DTW) finds the optimal alignment between two time series by warping the time dimension, allowing for more robust similarity comparisons when patterns are similar but shifted, stretched, or compressed in time.

### Implementation

```python
from refactor.analysis_modules.similarity import calculate_pairwise_dtw_distance, plot_dtw_distance_heatmap

# Calculate pairwise DTW distance matrix
dtw_matrix = calculate_pairwise_dtw_distance(
    data_df,           # DataFrame containing time series data
    time_col,          # Name of the column representing time
    metric_col,        # Name of the column representing the metric to analyze
    group_col,         # Name of the column representing the groups (e.g., 'tenant')
    min_observations,  # Minimum observations required for a valid series
    normalize=True     # Whether to normalize DTW distance by path length
)

# Visualize the DTW distance matrix
plot_dtw_distance_heatmap(
    dtw_matrix,        # DataFrame containing the DTW distance values
    title,             # Title for the plot
    output_dir,        # Directory to save the plot
    filename,          # Base filename for the plot (without extension)
    cmap="viridis_r",  # Colormap for the heatmap (reversed to show smaller values darker)
    fmt=".2f",         # Format for cell annotations
    annot=True,        # Whether to show annotations
    tables_dir         # Optional: directory to save the CSV table
)
```

### Interpretation

- Lower values indicate higher similarity between time series
- DTW distances are robust to time shifts and speed variations
- Particularly useful for detecting similar patterns regardless of phase shifts
- Z-score normalization is applied to focus on pattern shapes rather than magnitudes

## Using Similarity Analysis in Your Pipeline

### Basic Usage

To perform similarity analysis in your pipeline:

1. Load your time series data with proper time, metric, and group columns
2. Calculate the similarity/distance matrices using the appropriate functions
3. Visualize the results with heatmaps and export data tables as needed

Example:

```python
from refactor.analysis_modules.similarity import (
    calculate_pairwise_distance_correlation, plot_distance_correlation_heatmap,
    calculate_pairwise_cosine_similarity, plot_cosine_similarity_heatmap,
    calculate_pairwise_dtw_distance, plot_dtw_distance_heatmap
)

# For each metric and experiment phase
for metric in metrics:
    for round_name, phase_name in experiment_phases:
        # Get data for this metric and phase
        phase_data = data[(data['round'] == round_name) & (data['phase'] == phase_name)]
        
        # Distance Correlation
        dcor_matrix = calculate_pairwise_distance_correlation(
            phase_data, 'timestamp', 'value', 'tenant', min_observations=10
        )
        
        plot_distance_correlation_heatmap(
            dcor_matrix,
            f"Distance Correlation: {metric} ({round_name}, {phase_name})",
            "output/plots/similarity/distance_correlation",
            f"{metric}_{round_name}_{phase_name}_distance_correlation_heatmap",
            tables_dir="output/tables/similarity/distance_correlation"
        )
        
        # Cosine Similarity
        cosine_matrix = calculate_pairwise_cosine_similarity(
            phase_data, 'timestamp', 'value', 'tenant', min_observations=10
        )
        
        plot_cosine_similarity_heatmap(
            cosine_matrix,
            f"Cosine Similarity: {metric} ({round_name}, {phase_name})",
            "output/plots/similarity/cosine_similarity", 
            f"{metric}_{round_name}_{phase_name}_cosine_similarity_heatmap",
            tables_dir="output/tables/similarity/cosine_similarity"
        )
        
        # Dynamic Time Warping
        dtw_matrix = calculate_pairwise_dtw_distance(
            phase_data, 'timestamp', 'value', 'tenant', min_observations=10, normalize=True
        )
        
        plot_dtw_distance_heatmap(
            dtw_matrix,
            f"DTW Distance: {metric} ({round_name}, {phase_name})",
            "output/plots/similarity/dtw",
            f"{metric}_{round_name}_{phase_name}_dtw_distance_heatmap", 
            tables_dir="output/tables/similarity/dtw"
        )
```

### Command Line Usage

The similarity analyses are automatically run when using the main script with appropriate flags:

```bash
python -m refactor.new_main --data-dir your-data-dir --output-dir output-dir --dcor --cosine-sim --dtw
```

Available flags:
- `--dcor`: Run Distance Correlation analysis
- `--min-obs-dcor`: Minimum observations for dCor (default: 10)
- `--cosine-sim`: Run Cosine Similarity analysis
- `--min-obs-cosine`: Minimum observations for Cosine Similarity (default: 10)
- `--dtw`: Run Dynamic Time Warping analysis
- `--min-obs-dtw`: Minimum observations for DTW (default: 10)
- `--normalize-dtw`: Normalize DTW distance by path length (default: True)

### Best Practices

1. **Ensure sufficient observations**: Use an appropriate minimum number of observations for reliable results
2. **Normalize DTW distances**: Always normalize DTW distances when comparing series of different lengths
3. **Use Z-score normalization**: DTW implementation automatically Z-normalizes series to focus on patterns rather than magnitude
4. **Close figures after creation**: To avoid memory issues, especially when processing many metrics/phases
5. **Interpret results in context**: Different similarity measures have different strengths; combine insights from multiple methods
