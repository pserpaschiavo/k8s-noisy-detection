import pandas as pd
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
import numpy as np

def perform_pca(df: pd.DataFrame, n_components: int | float | None = None, variance_threshold: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray | None]:
    """
    Performs Principal Component Analysis (PCA) on the given DataFrame.

    Args:
        df: DataFrame with features to analyze. Assumes columns are features and rows are samples.
            Must not contain non-numeric data or NaN values.
        n_components: Number of principal components to keep.
                      If None, all components are kept.
                      If float (e.g., 0.95), it selects the number of components such that the amount
                      of variance that needs to be explained is greater than the percentage specified.
        variance_threshold: Alternative to n_components. If set, n_components will be determined
                            by the number of components needed to explain this cumulative variance.
                            (e.g., 0.95 for 95% variance). Overrides n_components if both are set and
                            n_components is an int.

    Returns:
        A tuple containing:
        - pca_df: DataFrame with the principal components.
        - components_df: DataFrame with the principal components (loadings).
        - explained_variance_ratio: Explained variance ratio for each component.
    """
    if df.isnull().values.any():
        raise ValueError("Input DataFrame for PCA contains NaN values. Please handle them before analysis.")
    if not all(df.dtypes.apply(pd.api.types.is_numeric_dtype)):
        raise ValueError("Input DataFrame for PCA contains non-numeric columns. Please ensure all data is numeric.")

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    if variance_threshold is not None and 0 < variance_threshold < 1:
        pca_temp = PCA()
        pca_temp.fit(scaled_data)
        cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components_for_variance = np.argmax(cumulative_variance >= variance_threshold) + 1
        print(f"PCA: Selecting {n_components_for_variance} components to explain at least {variance_threshold*100}% of variance.")
        pca = PCA(n_components=n_components_for_variance)
    elif isinstance(n_components, float) and 0 < n_components < 1:
        print(f"PCA: Selecting components to explain at least {n_components*100}% of variance.")
        pca = PCA(n_components=n_components)
    elif isinstance(n_components, int) and n_components > 0:
        if n_components > df.shape[1]:
            print(f"Warning: n_components ({n_components}) is greater than the number of features ({df.shape[1]}). Setting n_components to {df.shape[1]}.")
            n_components = df.shape[1]
        pca = PCA(n_components=n_components)
    else:
        pca = PCA() # Keep all components

    principal_components = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(data=principal_components, 
                          columns=[f'PC{i+1}' for i in range(principal_components.shape[1])],
                          index=df.index)
    
    components_df = pd.DataFrame(pca.components_, 
                                 columns=df.columns, 
                                 index=[f'PC{i+1}' for i in range(pca.n_components_)])
    
    explained_variance_ratio = pca.explained_variance_ratio_
    
    print(f"PCA: Explained variance by component: {explained_variance_ratio}")
    print(f"PCA: Total explained variance by {pca.n_components_} components: {np.sum(explained_variance_ratio):.4f}")

    return pca_df, components_df, explained_variance_ratio

def perform_ica(df: pd.DataFrame, n_components: int | None = None, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs Independent Component Analysis (ICA) on the given DataFrame.

    Args:
        df: DataFrame with features to analyze. Assumes columns are features and rows are samples.
            Must not contain non-numeric data or NaN values.
        n_components: Number of independent components to estimate.
                      If None, it defaults to the number of features.
        random_state: Seed for reproducibility.

    Returns:
        A tuple containing:
        - ica_df: DataFrame with the independent components.
        - components_df: DataFrame representing the estimated unmixing matrix.
    """
    if df.isnull().values.any():
        raise ValueError("Input DataFrame for ICA contains NaN values. Please handle them before analysis.")
    if not all(df.dtypes.apply(pd.api.types.is_numeric_dtype)):
        raise ValueError("Input DataFrame for ICA contains non-numeric columns. Please ensure all data is numeric.")

    # Standardize the data (ICA often benefits from this, similar to PCA)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    if n_components is None:
        n_components = scaled_data.shape[1]
    elif n_components > scaled_data.shape[1]:
        print(f"Warning: n_components ({n_components}) for ICA is greater than the number of features ({scaled_data.shape[1]}). Setting n_components to {scaled_data.shape[1]}.")
        n_components = scaled_data.shape[1]

    ica = FastICA(n_components=n_components, random_state=random_state, whiten='unit-variance', max_iter=500) # Added whiten and max_iter
    independent_components = ica.fit_transform(scaled_data) # Use fit_transform
    
    ica_df = pd.DataFrame(data=independent_components, 
                          columns=[f'IC{i+1}' for i in range(independent_components.shape[1])],
                          index=df.index)
    
    # The unmixing matrix (components_) allows to project the original data to the independent components.
    # If you want the mixing matrix, it's ica.mixing_
    components_df = pd.DataFrame(ica.components_, 
                                 columns=df.columns, 
                                 index=[f'IC{i+1}' for i in range(ica.n_components)]) # Changed ica.n_components_ to ica.n_components
    
    print(f"ICA: Estimated {ica.n_components} independent components.") # Changed ica.n_components_ to ica.n_components

    return ica_df, components_df

def get_top_features_per_component(components_df: pd.DataFrame, n_top_features: int = 5) -> pd.DataFrame:
    """
    Extracts the top N features for each component based on absolute coefficient values.

    Args:
        components_df: DataFrame where rows are components (e.g., PC1, IC1)
                       and columns are original feature names.
        n_top_features: Number of top features to extract for each component.

    Returns:
        A DataFrame with columns: ['Component', 'Rank', 'Feature', 'Coefficient']
    """
    top_features_list = []
    if components_df is None or components_df.empty:
        print("Warning: Components DataFrame is empty or None in get_top_features_per_component. Returning empty DataFrame.")
        return pd.DataFrame(top_features_list, columns=['Component', 'Rank', 'Feature', 'Coefficient'])

    for component_name, series in components_df.iterrows():
        # Sort features by absolute coefficient value in descending order
        # Ensure series is not all NaN or empty
        if series.empty or series.isnull().all():
            print(f"Warning: Component {component_name} has empty or all-NaN data. Skipping.")
            continue
            
        top_n = series.abs().sort_values(ascending=False).head(n_top_features)
        rank = 1
        for feature_name in top_n.index: # Iterate over index of top_n to get feature names
            original_coeff = series[feature_name]
            top_features_list.append({
                'Component': component_name,
                'Rank': rank,
                'Feature': feature_name,
                'Coefficient': original_coeff
            })
            rank += 1
    return pd.DataFrame(top_features_list)

# Example Usage (optional, for testing)
if __name__ == '__main__':
    # Create a sample DataFrame
    data_sample = {
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100) * 2,
        'feature3': np.random.rand(100) - 0.5,
        'feature4': np.random.rand(100) * 5,
        'tenant': [f'tenant_{i%3}' for i in range(100)] # Example non-numeric column
    }
    sample_df_full = pd.DataFrame(data_sample)
    sample_df_numeric = sample_df_full[['feature1', 'feature2', 'feature3', 'feature4']]
    
    print("Running PCA example...")
    try:
        # Test with n_components as int
        pca_results_int, pca_components_int, pca_variance_int = perform_pca(sample_df_numeric.copy(), n_components=2)
        print("\\nPCA Results (n_components=2):")
        print(pca_results_int.head())
        print("\\nPCA Components (n_components=2):")
        print(pca_components_int)
        print("\\nPCA Explained Variance (n_components=2):")
        print(pca_variance_int)

        # Test with n_components as float (variance explained)
        pca_results_float, pca_components_float, pca_variance_float = perform_pca(sample_df_numeric.copy(), n_components=0.90)
        print("\\nPCA Results (n_components for 90% variance):")
        print(pca_results_float.head())
        
        # Test with variance_threshold
        pca_results_thresh, pca_components_thresh, pca_variance_thresh = perform_pca(sample_df_numeric.copy(), variance_threshold=0.80)
        print("\\nPCA Results (variance_threshold=0.80):")
        print(pca_results_thresh.head())

        # Test with all components
        pca_results_all, _, _ = perform_pca(sample_df_numeric.copy())
        print(f"\\nPCA Results (all components - {pca_results_all.shape[1]}):")
        print(pca_results_all.head())

    except ValueError as e:
        print(f"PCA Error: {e}")

    print("\\n" + "="*30 + "\\n")

    print("Running ICA example...")
    try:
        ica_results, ica_components = perform_ica(sample_df_numeric.copy(), n_components=3, random_state=42)
        print("\\nICA Results (n_components=3):")
        print(ica_results.head())
        print("\\nICA Components (n_components=3):")
        print(ica_components)

        ica_results_all, _ = perform_ica(sample_df_numeric.copy(), random_state=42) # All components
        print(f"\\nICA Results (all components - {ica_results_all.shape[1]}):")
        print(ica_results_all.head())

    except ValueError as e:
        print(f"ICA Error: {e}")

    # Example of how to handle non-numeric data or NaNs before calling
    # sample_df_full_cleaned = sample_df_full.select_dtypes(include=np.number).dropna()
    # if not sample_df_full_cleaned.empty:
    #     pca_results_cleaned, _, _ = perform_pca(sample_df_full_cleaned)
    #     print("\\nPCA Results (cleaned data):")
    #     print(pca_results_cleaned.head())
    # else:
    #     print("\\nCleaned DataFrame is empty, skipping PCA on cleaned data.")
