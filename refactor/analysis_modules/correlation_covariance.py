import pandas as pd
import numpy as np

# Moved from pipeline.analysis.tenant_analysis
def calculate_correlation_matrix(metrics_dict, tenants=None, round_name='round-1', noisy_tenant=None, method='pearson'):
    """
    Calcula uma matriz de correlação entre métricas de diferentes tenants.
    
    Args:
        metrics_dict (dict): Dicionário com DataFrames para cada métrica
        tenants (list): Lista de tenants a incluir (None = todos)
        round_name (str): Round a ser analisado
        noisy_tenant (str): Tenant específico que gera ruído (por padrão: DEFAULT_NOISY_TENANT da configuração)
        method (str): Método de correlação a ser usado ('pearson', 'spearman', 'kendall'). Padrão: 'pearson'.
        
    Returns:
        DataFrame: Matriz de correlação entre métricas dos tenants
    """
    from pipeline.config import DEFAULT_NOISY_TENANT # TODO: Refactor to remove this direct import or pass as arg
    
    # Determinar qual é o tenant gerador de ruído
    noisy_tenant = noisy_tenant if noisy_tenant else DEFAULT_NOISY_TENANT
    
    # Preparar dados para correlação
    correlation_data = {}
    
    for metric_name, metric_df in metrics_dict.items():
        # Filtrar pelo round especificado
        round_df = metric_df[metric_df['round'] == round_name]
        
        if tenants:
            round_df = round_df[round_df['tenant'].isin(tenants)]
        
        # Verificar se o tenant gerador de ruído está presente
        has_noisy_tenant = noisy_tenant in round_df['tenant'].unique()
        
        # Pivotar para ter uma coluna para cada tenant
        pivot = round_df.pivot_table(
            index='datetime',
            columns='tenant',
            values='value'
        )
        
        # Preencher valores NaN com 0
        pivot.fillna(0, inplace=True)
        
        # Garantir que tenant gerador de ruído esteja presente para consistência nas análises
        if not has_noisy_tenant and noisy_tenant not in pivot.columns:
            pivot[noisy_tenant] = 0  # Adicionar tenant gerador de ruído com valores zero
        
        # Adicionar ao dicionário com prefixo da métrica
        for tenant in pivot.columns:
            correlation_data[f"{metric_name}_{tenant}"] = pivot[tenant]
    
    # Criar DataFrame com todas as séries
    corr_df = pd.DataFrame(correlation_data)
    
    # Calcular correlação
    correlation_matrix = corr_df.corr(method=method)
    
    return correlation_matrix

# Updated function
def calculate_inter_tenant_correlation_per_metric(
    metric_df_single_round: pd.DataFrame, 
    method: str = 'pearson', 
    time_col: str = 'timestamp'  # Default to 'timestamp', assuming it's datetime
) -> pd.DataFrame:
    """
    Calcula a matriz de correlação inter-tenant para uma única métrica e um único round.
    O DataFrame de entrada deve estar em formato longo.

    Args:
        metric_df_single_round (pd.DataFrame): DataFrame contendo dados de uma métrica
                                                 para múltiplos tenants em um único round.
                                                 Deve conter colunas [time_col], 'tenant', 'value'.
        method (str): Método de correlação ('pearson', 'spearman', 'kendall').
        time_col (str): Nome da coluna de tempo a ser usada como índice para pivotar.
    Returns:
        pd.DataFrame: Matriz de correlação inter-tenant. Vazia se dados insuficientes.
    """
    required_cols = {time_col, 'tenant', 'value'}
    if not required_cols.issubset(metric_df_single_round.columns):
        print(f"Error: DataFrame for correlation must contain columns: {required_cols}. Found: {metric_df_single_round.columns}")
        return pd.DataFrame()

    try:
        pivot_df = metric_df_single_round.pivot_table(
            index=time_col,
            columns='tenant',
            values='value'
        )
    except Exception as e:
        print(f"Error during pivot_table in calculate_inter_tenant_correlation_per_metric: {e}")
        return pd.DataFrame()

    # Handle potential NaNs from pivoting
    pivot_df = pivot_df.ffill().bfill()
    
    pivot_df = pivot_df.dropna(axis=1, how='all') # Drop tenants (columns) that are entirely NaN
    pivot_df = pivot_df.dropna(axis=0, how='any')  # Drop timestamps (rows) where any remaining tenant still has NaN

    if pivot_df.shape[1] < 2:
        # print(f"Skipping correlation: Not enough tenant data after pivoting and cleaning (need at least 2 tenants). Found {pivot_df.shape[1]}.")
        return pd.DataFrame()
    if pivot_df.shape[0] < 2:
        # print(f"Skipping correlation: Not enough data points after pivoting and cleaning (need at least 2 data points). Found {pivot_df.shape[0]}.")
        return pd.DataFrame()

    return pivot_df.corr(method=method)

# Moved from pipeline.analysis.tenant_analysis
def calculate_inter_tenant_covariance_per_metric(metric_df_single_round: pd.DataFrame, time_col: str = 'timestamp') -> pd.DataFrame:
    """
    Calcula a matriz de covariância inter-tenant para uma única métrica e um único round.
    O DataFrame de entrada deve estar em formato longo.

    Args:
        metric_df_single_round (pd.DataFrame): DataFrame contendo dados de uma métrica
                                                 para múltiplos tenants em um único round.
                                                 Deve conter colunas [time_col], 'tenant', 'value'.
        time_col (str): Nome da coluna de tempo a ser usada como índice para pivotar.

    Returns:
        pd.DataFrame: Matriz de covariância inter-tenant. Vazia se dados insuficientes.
    """
    required_cols = {time_col, 'tenant', 'value'}
    if not required_cols.issubset(metric_df_single_round.columns):
        print(f"Error: DataFrame for covariance must contain columns: {required_cols}. Found: {metric_df_single_round.columns}")
        return pd.DataFrame()
    
    try:
        pivot_df = metric_df_single_round.pivot_table(
            index=time_col,
            columns='tenant',
            values='value'
        )
    except Exception as e:
        print(f"Error during pivot_table in calculate_inter_tenant_covariance_per_metric: {e}")
        return pd.DataFrame()

    pivot_df = pivot_df.ffill().bfill()
    pivot_df = pivot_df.dropna(axis=1, how='all')
    pivot_df = pivot_df.dropna(axis=0, how='any')

    if pivot_df.shape[1] < 2:
        return pd.DataFrame()
    if pivot_df.shape[0] < 2:
        return pd.DataFrame()
        
    return pivot_df.cov()

# Moved from pipeline.analysis.advanced_analysis
def calculate_covariance_matrix(metrics_dict, tenants=None, phase=None, round_name='round-1', correlation_method='pearson'):
    """
    Calcula uma matriz de covariância entre métricas de diferentes tenants.
    
    Args:
        metrics_dict (dict): Dicionário com DataFrames para cada métrica
        tenants (list): Lista de tenants a incluir (None = todos)
        phase (str): Fase específica para análise (None = todas)
        round_name (str): Round a ser analisado
        correlation_method (str): Método de correlação a ser usado para a matriz de correlação de comparação ('pearson', 'spearman', 'kendall'). Padrão: 'pearson'.
        
    Returns:
        DataFrame: Matriz de covariância entre métricas dos tenants
        DataFrame: Matriz de correlação (para comparação)
    """
    # Preparar dados para covariância
    covariance_data = {}
    
    for metric_name, metric_df in metrics_dict.items():
        # Filtrar pelo round especificado
        round_df = metric_df[metric_df['round'] == round_name]
        
        # Filtrar pela fase se especificada
        if phase:
            round_df = round_df[round_df['phase'] == phase]
            
        if tenants:
            round_df = round_df[round_df['tenant'].isin(tenants)]
        
        # Pivotar para ter uma coluna para cada tenant
        pivot = round_df.pivot_table(
            index='datetime',
            columns='tenant',
            values='value'
        )
        
        # Adicionar ao dicionário com prefixo da métrica
        for tenant in pivot.columns:
            covariance_data[f"{metric_name}_{tenant}"] = pivot[tenant]
    
    # Criar DataFrame com todas as séries
    cov_df = pd.DataFrame(covariance_data)
    
    # Calcular covariância
    covariance_matrix = cov_df.cov()
    
    # Calcular correlação para comparação
    correlation_matrix = cov_df.corr(method=correlation_method)
    
    return covariance_matrix, correlation_matrix

def calculate_cross_correlation(series1: pd.Series, series2: pd.Series, max_lag: int, method: str = 'pearson') -> pd.Series:
    """
    Calcula a Função de Correlação Cruzada (FCC) entre duas séries temporais.
    A correlação é calculada como corr(series1[t], series2[t-lag]).
    Um lag positivo significa que series2 está atrasada em relação a series1 (series1 lidera series2).
    Um lag negativo significa que series2 está adiantada em relação a series1 (series2 lidera series1).

    Args:
        series1 (pd.Series): A primeira série temporal (referência).
        series2 (pd.Series): A segunda série temporal (a ser defasada).
        max_lag (int): O número máximo de lags (defasagens) a serem testados.
                       O range de lags será de -max_lag a +max_lag.
        method (str): Método de correlação a ser usado ('pearson', 'spearman', 'kendall'). Padrão: 'pearson'.

    Returns:
        pd.Series: Uma série contendo os coeficientes de correlação para cada lag.
                   O índice da série representa os lags.
    """
    if not isinstance(series1, pd.Series) or not isinstance(series2, pd.Series):
        raise TypeError("As entradas series1 e series2 devem ser objetos pandas Series.")

    if len(series1) != len(series2):
        raise ValueError("As séries temporais devem ter o mesmo comprimento.")

    if series1.empty or series2.empty:
        idx = pd.RangeIndex(start=-max_lag, stop=max_lag + 1, name='Lag')
        return pd.Series(dtype=float, index=idx, name='Cross-Correlation')

    lags = range(-max_lag, max_lag + 1)
    correlations = []

    # Ensure series are float for correlation calculation, NaNs are handled by .corr()
    s1 = series1.astype(float)
    s2 = series2.astype(float)

    for lag in lags:
        # series2 is shifted by 'lag'.
        # if lag > 0, series2 is shifted "forward" (values from past appear later)
        # series1[t] vs series2[t-lag]
        shifted_s2 = s2.shift(lag)
        correlation = s1.corr(shifted_s2, method=method)
        correlations.append(correlation)
    
    return pd.Series(data=correlations, index=pd.Index(lags, name='Lag'), name='Cross-Correlation')
