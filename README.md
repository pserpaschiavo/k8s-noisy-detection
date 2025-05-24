# Kubernetes Noisy Neighbor Detection

This project provides a comprehensive pipeline for analyzing and detecting noisy neighbors in Kubernetes environments through time series analysis.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/phil/k8s-noisy-detection.git
cd k8s-noisy-detection

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Basic Usage

```bash
# Run analysis on demo data
python -m src.main --data-dir demo-data/demo-experiment-1-round --output-dir output

# Run with specific metrics
python -m src.main --data-dir demo-data/demo-experiment-1-round --output-dir output --selected-metrics cpu_usage memory_usage
```

## 📁 Project Structure

```
k8s-noisy-detection/
├── src/                          # Main source code
│   ├── config.py                 # Configuration settings
│   ├── main.py                   # Main analysis pipeline
│   ├── analysis/                 # Analysis modules
│   │   ├── causality.py          # Causal analysis
│   │   ├── correlation_covariance.py
│   │   ├── descriptive_statistics.py
│   │   ├── multivariate.py       # PCA, ICA, t-SNE
│   │   ├── root_cause.py         # Root cause analysis
│   │   └── similarity.py         # Similarity metrics
│   ├── data/                     # Data handling
│   │   ├── loader.py             # Data loading utilities
│   │   ├── normalization.py      # Data normalization
│   │   └── io_utils.py           # I/O operations
│   ├── utils/                    # Utilities
│   │   ├── common.py             # Common imports and utilities
│   │   └── figure_management.py  # Plot management
│   └── visualization/            # Visualization
│       └── plots.py              # All plotting functions
├── tests/                        # Test suite
├── demo-data/                    # Sample data
├── docs/                         # Documentation
├── requirements.txt              # Dependencies
├── setup.py                      # Package setup
└── README.md
```

## 🔧 Analysis Features

- **Data Loading**: Load and preprocess time series data from Kubernetes experiments
- **Descriptive Statistics**: Calculate and visualize basic statistics for time series data
- **Correlation Analysis**: Perform correlation analysis between different metrics and tenants
- **Covariance Analysis**: Analyze covariance patterns in time series data
- **Multivariate Analysis**: Apply PCA, ICA, and t-SNE for dimensionality reduction and pattern discovery
- **Similarity Analysis**: Measure similarity between time series using multiple techniques
- **Causal Analysis**: Explore causal relationships using SEM and other statistical methods
- **Root Cause Analysis**: Identify potential causes of performance issues

## 📊 Time Series Similarity Methods

The project implements three main techniques for measuring similarity between time series:

### 1. Distance Correlation (dCor)

Distance Correlation measures dependence between variables, capable of detecting nonlinear relationships. It is equal to zero if and only if the variables are statistically independent. The implementation uses the `dcor` Python package.

Key characteristics:
- Detects both linear and nonlinear dependencies
- Values range from 0 (independent) to 1 (strong dependency)
- Does not assume normal distribution or linearity

### 2. Cosine Similarity

Cosine Similarity measures the cosine of the angle between two vectors, providing a similarity metric independent of magnitude, focusing on orientation in the feature space.

Key characteristics:
- Values range from -1 (exactly opposite) to 1 (exactly the same)
- Insensitive to scaling, focuses on directional similarity
- Efficient for sparse, high-dimensional data

### 3. Dynamic Time Warping (DTW)

DTW finds the optimal alignment between two time series by warping the time dimension to minimize the distance between corresponding points. This allows comparing patterns even when they are shifted, stretched, or compressed in time.

Key characteristics:
- Handles time series of different lengths
- Robust to time shifts and speed variations
- Returns a distance measure (lower values indicate higher similarity)
- Uses the `tslearn` implementation for efficient computation

## Usage

Run the analysis using the `new_main.py` script with appropriate arguments:

```bash
python refactor/new_main.py --data-dir demo-data/demo-experiment-3-rounds --dtw --cosine-sim --dcor
```

### Command Line Arguments

Common arguments:
- `--data-dir`: Directory containing experiment data
- `--output-dir`: Directory to save analysis results
- `--metrics`: Specific metrics to analyze
- `--tenants`: Specific tenants to analyze

Similarity analysis arguments:
- `--dcor`: Enable Distance Correlation analysis
- `--min-obs-dcor`: Minimum observations required for dCor (default: 10)
- `--cosine-sim`: Enable Cosine Similarity analysis
- `--min-obs-cosine`: Minimum observations for Cosine Similarity (default: 10)
- `--dtw`: Enable Dynamic Time Warping analysis
- `--min-obs-dtw`: Minimum observations for DTW (default: 10)
- `--normalize-dtw`: Normalize DTW distance by path length (default: True)

## Output

Analysis results are organized in the following directories:
- `output/plots/`: Visualization plots
- `output/tables/`: Data tables
- Subdirectories for each analysis type (e.g., `similarity/dtw/`)

## Dependencies

Key dependencies include:
- pandas, numpy, matplotlib, seaborn: Core data analysis and visualization
- dcor: Distance correlation calculation
- tslearn: Dynamic Time Warping implementation
- scikit-learn: Multivariate analysis (PCA, ICA)
- statsmodels, semopy: Statistical and causal modeling

Install all requirements using:
```bash
pip install -r requirements.txt
```
