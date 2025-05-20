"""
Módulo para análise de causalidade entre tenants.

Este módulo contém funções para investigar e quantificar as relações de causa e efeito
entre o comportamento de diferentes tenants no ambiente Kubernetes, especialmente
em cenários de noisy neighbor.
"""
import pandas as pd
from typing import Dict, List, Tuple, Optional
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np

def calculate_causal_impact_between_tenants(
    metric_df: pd.DataFrame,
    source_tenant: str,
    target_tenant: str,
    metric_name: str,
    time_column: str = 'experiment_elapsed_seconds',
    value_column: str = 'value',
    control_tenants: Optional[List[str]] = None,
    max_lag: int = 5,
    verbose: bool = False,
    **kwargs
) -> Dict:
    """
    Calcula o impacto causal de um tenant (source) em outro tenant (target)
    para uma métrica específica usando Causalidade de Granger.

    Args:
        metric_df (pd.DataFrame): DataFrame contendo as séries temporais das métricas.
        source_tenant (str): Identificador do tenant de origem.
        target_tenant (str): Identificador do tenant de destino.
        metric_name (str): Nome da métrica.
        time_column (str): Nome da coluna de tempo.
        value_column (str): Nome da coluna de valor da métrica.
        control_tenants (Optional[List[str]]): Não utilizado nesta implementação.
        max_lag (int): O número máximo de lags para testar na Causalidade de Granger.
        verbose (bool): Se True, imprime os resultados do teste de Granger.
        **kwargs: Argumentos adicionais (não utilizados atualmente).

    Returns:
        Dict: Um dicionário contendo os resultados da análise de Causalidade de Granger.
              Inclui 'min_p_value' e 'best_lag' com o menor p-valor.
    """
    if control_tenants:
        print("Warning: control_tenants are not used in this Granger causality implementation.")

    series_target, series_source = prepare_data_for_granger_causality(
        metric_df,
        target_tenant, # First argument to prepare_data is the 'dependent' variable (effect)
        source_tenant, # Second argument is the 'independent' variable (cause)
        metric_name,
        time_column,
        value_column
    )

    if series_source.empty or series_target.empty:
        return {
            "source_tenant": source_tenant,
            "target_tenant": target_tenant,
            "metric_name": metric_name,
            "error": "Uma ou ambas as séries estão vazias após o preparo.",
            "min_p_value": None,
            "best_lag": None,
            "all_p_values": {}
        }

    # Granger causality test requires a DataFrame with both series
    data_for_granger = pd.DataFrame({
        'target': series_target,
        'source': series_source
    })
    data_for_granger = data_for_granger.dropna() # Ensure no NaNs before test

    if len(data_for_granger) < 3 * max_lag: # Heuristic: need enough data points
        return {
            "source_tenant": source_tenant,
            "target_tenant": target_tenant,
            "metric_name": metric_name,
            "error": f"Não há dados suficientes para o teste de Granger com max_lag={max_lag}. Dados disponíveis: {len(data_for_granger)}",
            "min_p_value": None,
            "best_lag": None,
            "all_p_values": {}
        }

    try:
        # The first variable in the DataFrame is the one being caused (target/effect)
        # The second variable is the one causing (source/cause)
        # So, we test if 'source' Granger-causes 'target'
        results = grangercausalitytests(data_for_granger[['target', 'source']], maxlag=max_lag, verbose=verbose)
        
        min_p_value = 1.0
        best_lag = -1
        all_p_values = {}

        for lag in results:
            # Test results are a tuple; p-value is typically the second element of the first test ('ssr_ftest')
            # Example: results[lag][0] is a dict like {'ssr_ftest': (f_stat, p_val, df_num, df_den), ...}
            p_value = results[lag][0]['ssr_ftest'][1]
            all_p_values[lag] = p_value
            if p_value < min_p_value:
                min_p_value = p_value
                best_lag = lag
        
        return {
            "source_tenant": source_tenant,
            "target_tenant": target_tenant,
            "metric_name": metric_name,
            "min_p_value": min_p_value,
            "best_lag": best_lag,
            "all_p_values": all_p_values,
            "message": "Teste de Causalidade de Granger concluído."
        }
    except Exception as e:
        # Catch specific exceptions if possible, e.g., LinAlgError for singular matrix
        return {
            "source_tenant": source_tenant,
            "target_tenant": target_tenant,
            "metric_name": metric_name,
            "error": f"Erro ao executar o teste de Granger: {str(e)}",
            "min_p_value": None,
            "best_lag": None,
            "all_p_values": {}
        }

def identify_causal_chains(
    metric_df: pd.DataFrame,
    all_tenants: List[str],
    metric_names: List[str],
    causality_threshold: float = 0.05,
    max_lag_granger: int = 5,
    verbose_granger: bool = False,
    **kwargs
) -> List[Tuple[str, str, str, float, int]]:
    """
    Identifica cadeias de relações causais entre múltiplos tenants para um conjunto de métricas,
    usando a Causalidade de Granger.

    Args:
        metric_df (pd.DataFrame): DataFrame com todas as séries temporais.
        all_tenants (List[str]): Lista de todos os tenants a serem considerados.
        metric_names (List[str]): Lista das métricas para as quais se busca causalidade.
        causality_threshold (float): Limiar (p-valor) para considerar uma relação causal como significativa.
        max_lag_granger (int): Lag máximo para o teste de Granger.
        verbose_granger (bool): Se True, imprime os resultados do teste de Granger.
        **kwargs: Argumentos adicionais para `calculate_causal_impact_between_tenants`.

    Returns:
        List[Tuple[str, str, str, float, int]]: Uma lista de tuplas, onde cada tupla representa
                                           um link causal significativo:
                                           (source_tenant, target_tenant, metric_name, p_value, best_lag)
    """
    causal_links = []
    if len(all_tenants) < 2:
        print("Pelo menos dois tenants são necessários para análise de causalidade.")
        return causal_links

    for metric in metric_names:
        print(f"Analisando causalidade para a métrica: {metric}")
        for i in range(len(all_tenants)):
            for j in range(len(all_tenants)):
                if i == j:
                    continue # Não testar causalidade de um tenant para ele mesmo

                source_tenant = all_tenants[i]
                target_tenant = all_tenants[j]

                print(f"  Testando: {source_tenant} -> {target_tenant} para {metric}")
                
                # Passar max_lag e verbose para a função de cálculo
                causality_result = calculate_causal_impact_between_tenants(
                    metric_df=metric_df,
                    source_tenant=source_tenant,
                    target_tenant=target_tenant,
                    metric_name=metric,
                    max_lag=max_lag_granger,
                    verbose=verbose_granger,
                    **kwargs 
                )

                if "error" in causality_result:
                    print(f"    Erro no teste de causalidade para {source_tenant} -> {target_tenant} ({metric}): {causality_result['error']}")
                    continue

                p_value = causality_result.get("min_p_value")
                best_lag = causality_result.get("best_lag")

                if p_value is not None and p_value < causality_threshold:
                    print(f"    Causalidade significativa encontrada: {source_tenant} -> {target_tenant} para {metric} (p-valor: {p_value:.4f}, lag: {best_lag})")
                    causal_links.append((source_tenant, target_tenant, metric, p_value, best_lag))
                elif p_value is not None:
                    print(f"    Causalidade não significativa: {source_tenant} -> {target_tenant} para {metric} (p-valor: {p_value:.4f})")
                else:
                    print(f"    Resultado do teste de causalidade inválido para {source_tenant} -> {target_tenant} ({metric}).")


    if not causal_links:
        print("Nenhum link causal significativo encontrado com os critérios atuais.")
    else:
        print(f"Total de links causais significativos encontrados: {len(causal_links)}")
        
    return causal_links

def visualize_causal_graph(
    causal_links: List[Tuple[str, str, str, float, int]], # Added best_lag to tuple
    output_path: Optional[str] = None,
    metric_colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (15, 12),
    node_size: int = 4500,
    font_size: int = 16, # MODIFIED default font_size from 14 to 16
    arrow_size: int = 20,
    layout_type: str = 'spring', # e.g., spring, circular, kamada_kawai
    **kwargs
) -> object:
    """
    Gera uma visualização (grafo) das relações causais identificadas.

    Args:
        causal_links (List[Tuple[str, str, str, float, int]]):
            Lista de links causais. Cada tupla: 
            (source_tenant, target_tenant, metric_name, p_value, best_lag).
        output_path (Optional[str]): Caminho para salvar a imagem do grafo.
        metric_colors (Optional[Dict[str, str]]): Dicionário mapeando nomes de métricas para cores.
        figsize (Tuple[int, int]): Tamanho da figura do plot.
        node_size (int): Tamanho dos nós no grafo.
        font_size (int): Tamanho da fonte para labels.
        arrow_size (int): Tamanho das setas das arestas.
        layout_type (str): Tipo de layout do grafo (ex: 'spring', 'circular', 'kamada_kawai', 'shell').
        **kwargs: Argumentos adicionais para a biblioteca de visualização (NetworkX).

    Returns:
        object: Objeto da figura matplotlib gerado, ou None se salvo em arquivo.
    """
    if not causal_links:
        print("Nenhum link causal para visualizar.")
        return None

    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        print("Por favor, instale networkx e matplotlib para visualizar o grafo: pip install networkx matplotlib")
        return {"error": "Dependências ausentes: networkx, matplotlib"}

    G = nx.DiGraph()
    
    # Adicionar nós primeiro para garantir que todos os tenants estejam presentes
    all_tenants_in_links = set()
    for source, target, _, _, _ in causal_links:
        all_tenants_in_links.add(source)
        all_tenants_in_links.add(target)
    for tenant in all_tenants_in_links:
        G.add_node(tenant)

    # Mapear métricas para cores se não fornecido
    if metric_colors is None:
        unique_metrics = sorted(list(set(link[2] for link in causal_links)))
        # Gerar cores distintas (pode ser melhorado com uma paleta de cores)
        default_colors = plt.cm.get_cmap('tab10', len(unique_metrics) if len(unique_metrics) > 0 else 1)
        metric_colors = {metric: default_colors(i) for i, metric in enumerate(unique_metrics)}
    
    # Adicionar arestas com atributos
    edge_labels = {}
    for source, target, metric, p_value, lag in causal_links:
        G.add_edge(source, target, label=f"{metric}\np={p_value:.2f}, lag={lag}", 
                   color=metric_colors.get(metric, 'gray'), weight=(1-p_value), metric=metric)

    # Escolher layout
    if layout_type == 'spring':
        pos = nx.spring_layout(G, k=kwargs.get('k', 1.1), iterations=kwargs.get('iterations', 75), seed=kwargs.get('seed', 42)) # MODIFIED k
    elif layout_type == 'circular':
        pos = nx.circular_layout(G)
    elif layout_type == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G, weight='weight') # Usa o peso para o layout
    elif layout_type == 'shell':
        shells = None # Poderia ser [[tenant1, tenant2], [tenant3, tenant4]]
        if 'shells' in kwargs:
            shells = kwargs['shells']
        elif len(all_tenants_in_links) > 0:
            shells = [list(all_tenants_in_links)]
        if shells:
             pos = nx.shell_layout(G, nlist=shells)
        else:
            pos = nx.shell_layout(G) # Fallback se shells não puder ser determinado
    else:
        pos = nx.spring_layout(G, seed=42) # Default fallback

    plt.figure(figsize=figsize)
    
    edge_colors = [G[u][v]['color'] for u,v in G.edges()]
    
    nx.draw(G, pos, with_labels=True, node_size=node_size, node_color="skyblue", 
            font_size=font_size, font_weight='bold',
            arrows=True, arrowstyle='-|>', arrowsize=arrow_size,
            edge_color=edge_colors, width=2, # width é a espessura da linha da aresta
            connectionstyle='arc3,rad=0.1')

    current_edge_labels = {(u,v): d['label'] for u,v,d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=current_edge_labels, font_size=font_size-4, # MODIFIED font_size
                                 label_pos=0.35, # ADDED label_pos to shift labels
                                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.85))

    plt.title("Grafo de Causalidade Direcionada entre Tenants", fontsize=font_size + 4)
    
    if metric_colors:
        patches = [plt.plot([],[], marker="o", ms=10, ls="", mec=None, color=color, 
                        label=f"{metric}")[0]  for metric, color in metric_colors.items() if metric in [d['metric'] for u,v,d in G.edges(data=True)]]
        if patches:
            plt.legend(handles=patches, title="Métricas", loc="best", fontsize=font_size)

    plt.tight_layout()

    if output_path:
        try:
            plt.savefig(output_path, bbox_inches='tight', dpi=kwargs.get('dpi', 300))
            print(f"Grafo de causalidade salvo em {output_path}")
            plt.close() # Fechar a figura para liberar memória
            return None
        except Exception as e:
            print(f"Erro ao salvar o grafo: {e}")
            try:
                fig = plt.gcf()
                plt.show()
                return fig
            except Exception as e_show:
                 print(f"Erro ao exibir o grafo: {e_show}")
                 return {"error": f"Erro ao salvar e exibir grafo: {e}, {e_show}"}

    else:
        fig = plt.gcf()
        return fig

def prepare_data_for_granger_causality(
    metric_df: pd.DataFrame,
    tenant_dependent: str,
    tenant_independent: str,
    metric_name: str,
    time_column: str = 'experiment_elapsed_seconds',
    value_column: str = 'value'
) -> Tuple[pd.Series, pd.Series]:
    """
    Prepara as séries temporais de dois tenants para a análise de Causalidade de Granger.
    A primeira série retornada é a variável dependente (efeito).
    A segunda série retornada é a variável independente (causa).

    Args:
        metric_df (pd.DataFrame): DataFrame contendo as séries temporais.
        tenant_dependent (str): Identificador do tenant que sofre o efeito.
        tenant_independent (str): Identificador do tenant que causa o efeito.
        metric_name (str): Nome da métrica.
        time_column (str): Nome da coluna de tempo.
        value_column (str): Nome da coluna de valor.

    Returns:
        Tuple[pd.Series, pd.Series]: Duas séries temporais (valores da métrica para tenant_dependent e tenant_independent).
    """
    series_dependent = metric_df[
        (metric_df['tenant'] == tenant_dependent) & (metric_df['metric_name'] == metric_name)
    ].set_index(time_column)[value_column].sort_index()

    series_independent = metric_df[
        (metric_df['tenant'] == tenant_independent) & (metric_df['metric_name'] == metric_name)
    ].set_index(time_column)[value_column].sort_index()

    # Assegurar que as séries tenham o mesmo índice de tempo e não tenham NaNs
    aligned_series_dependent, aligned_series_independent = series_dependent.align(series_independent, join='inner')
    
    # Tratar NaNs que podem surgir do alinhamento ou já existir
    aligned_series_dependent = aligned_series_dependent.astype(np.float64).fillna(method='ffill').fillna(method='bfill')
    aligned_series_independent = aligned_series_independent.astype(np.float64).fillna(method='ffill').fillna(method='bfill')
    
    # Remover quaisquer NaNs restantes que não puderam ser preenchidos
    common_index = aligned_series_dependent.dropna().index.intersection(aligned_series_independent.dropna().index)
    aligned_series_dependent = aligned_series_dependent.loc[common_index]
    aligned_series_independent = aligned_series_independent.loc[common_index]

    return aligned_series_dependent, aligned_series_independent
