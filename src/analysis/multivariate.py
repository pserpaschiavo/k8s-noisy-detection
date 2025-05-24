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
    
    # Check for problematic values that could cause numerical issues
    if np.isnan(scaled_data).any() or np.isinf(scaled_data).any():
        print("Warning: Data contains NaN or Inf after standardization. Applying cleaning.")
        # Replace NaN and Inf with zeros
        scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=0.0, neginf=0.0)

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

def perform_ica(df: pd.DataFrame, n_components: int | None = None, random_state: int = 42, max_iter: int = 2000, tol: float = 1e-5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs Independent Component Analysis (ICA) on the given DataFrame.

    Args:
        df: DataFrame with features to analyze. Assumes columns are features and rows are samples.
            Must not contain non-numeric data or NaN values.
        n_components: Number of independent components to estimate.
                      If None, it defaults to the number of features.
        random_state: Seed for reproducibility.
        max_iter: Maximum number of iterations during algorithm execution.
        tol: Tolerance for convergence.

    Returns:
        A tuple containing:
        - ica_transformed_df: DataFrame with the independent components (sources).
        - ica_components_df: DataFrame representing the estimated unmixing matrix.
        - ica_mixing_df: DataFrame representing the estimated mixing matrix.
    """
    if df.isnull().values.any():
        raise ValueError("Input DataFrame for ICA contains NaN values. Please handle them before analysis.")
    if not all(df.dtypes.apply(pd.api.types.is_numeric_dtype)):
        raise ValueError("Input DataFrame for ICA contains non-numeric columns. Please ensure all data is numeric.")

    # Standardize the data (ICA often benefits from this, similar to PCA)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Check for problematic values that could cause numerical issues
    if np.isnan(scaled_data).any() or np.isinf(scaled_data).any():
        print("Warning: Data contains NaN or Inf after standardization. Applying cleaning.")
        # Replace NaN and Inf with zeros
        scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Add a small value to ensure numerical stability
    epsilon = 1e-10
    scaled_data = np.where(np.abs(scaled_data) < epsilon, epsilon, scaled_data)

    if n_components is None:
        n_components = scaled_data.shape[1]
    elif n_components > scaled_data.shape[1]:
        print(f"Warning: n_components ({n_components}) for ICA is greater than the number of features ({scaled_data.shape[1]}). Setting n_components to {scaled_data.shape[1]}.")
        n_components = scaled_data.shape[1]

    if n_components == 0:
        print("Warning: ICA n_components is 0. Cannot perform ICA. Returning empty DataFrames.")
        empty_df_indexed = pd.DataFrame(index=df.index)
        empty_df_features = pd.DataFrame(columns=df.columns)
        return empty_df_indexed, empty_df_features, empty_df_features

    ica = FastICA(
        n_components=n_components, 
        random_state=random_state, 
        whiten='unit-variance',  # Use unit-variance whitening (more stable)
        max_iter=max_iter,       # Use passed max_iter
        tol=tol                  # Use passed tol
    )
    
    try:
        independent_components = ica.fit_transform(scaled_data)
    except Exception as e:
        print(f"ICA fitting error: {str(e)}")
        empty_df = pd.DataFrame(index=df.index)
        empty_components = pd.DataFrame(columns=df.columns)
        return empty_df, empty_components, empty_components
    
    ica_transformed_df = pd.DataFrame(data=independent_components, 
                                      columns=[f'IC{i+1}' for i in range(independent_components.shape[1])],
                                      index=df.index)
    
    ica_components_df = pd.DataFrame(ica.components_, 
                                     columns=df.columns, 
                                     index=[f'IC{i+1}' for i in range(ica.n_components)])
    
    if hasattr(ica, 'mixing_') and ica.mixing_ is not None:
        ica_mixing_matrix = ica.mixing_
    else:
        print("Warning: ica.mixing_ not directly available. Attempting to compute from unmixing matrix.")
        try:
            ica_mixing_matrix = np.linalg.pinv(ica.components_)
        except np.linalg.LinAlgError as e:
            print(f"Error computing pseudo-inverse of ICA components: {e}. Mixing matrix will be empty.")
            ica_mixing_matrix = np.full((df.shape[1], independent_components.shape[1]), np.nan)

    ica_mixing_df = pd.DataFrame(ica_mixing_matrix,
                                 index=df.columns,
                                 columns=[f'IC{i+1}' for i in range(independent_components.shape[1])])
    
    print(f"ICA: Estimated {ica.n_components} independent components.")

    return ica_transformed_df, ica_components_df, ica_mixing_df

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
        if series.empty or series.isnull().all():
            print(f"Warning: Component {component_name} has empty or all-NaN data. Skipping.")
            continue
            
        top_n = series.abs().sort_values(ascending=False).head(n_top_features)
        rank = 1
        for feature_name in top_n.index:
            original_coeff = series[feature_name]
            top_features_list.append({
                'Component': component_name,
                'Rank': rank,
                'Feature': feature_name,
                'Coefficient': original_coeff
            })
            rank += 1
    return pd.DataFrame(top_features_list)

def perform_kpca(df: pd.DataFrame, n_components: int = None, kernel: str = 'rbf', gamma: float = None) -> pd.DataFrame:
    """
    Performs Kernel PCA on the given DataFrame.
    Returns a DataFrame of principal components.
    """
    from sklearn.decomposition import KernelPCA
    if df.isnull().values.any():
        raise ValueError("Input DataFrame for KPCA contains NaN values.")
    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, fit_inverse_transform=False)
    transformed = kpca.fit_transform(df)
    cols = [f'KPC{i+1}' for i in range(transformed.shape[1])]
    return pd.DataFrame(transformed, columns=cols, index=df.index)

def perform_tsne(df: pd.DataFrame, n_components: int = 2, perplexity: float = 30.0, random_state: int = 42) -> pd.DataFrame:
    """
    Performs t-SNE on the given DataFrame.
    Returns a DataFrame of embedded components.
    """
    from sklearn.manifold import TSNE
    if df.isnull().values.any():
        raise ValueError("Input DataFrame for t-SNE contains NaN values.")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    embedded = tsne.fit_transform(df)
    cols = [f'TSNE{i+1}' for i in range(embedded.shape[1])]
    return pd.DataFrame(embedded, columns=cols, index=df.index)

def perform_umap(df: pd.DataFrame, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42) -> pd.DataFrame:
    """
    Performs UMAP dimensionality reduction on the given DataFrame.
    Returns a DataFrame of embedded components.
    """
    try:
        import umap
    except ImportError:
        raise ImportError("UMAP requires the 'umap-learn' package. Please install it with pip install umap-learn.")
