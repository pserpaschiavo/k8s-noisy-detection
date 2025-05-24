import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import re

def list_available_tenants(experiment_dir):
    """
    Lista todos os tenants disponíveis no diretório do experimento.
    
    Args:
        experiment_dir (str): Caminho para o diretório do experimento
        
    Returns:
        list: Lista de nomes de tenants
    """
    tenants = set()
    
    # Procura por diretórios de tenants em todas as fases e rounds
    # Corrigido para iterar corretamente sobre a estrutura de diretórios
    for round_folder_name in os.listdir(experiment_dir):
        round_dir_path = os.path.join(experiment_dir, round_folder_name)
        if os.path.isdir(round_dir_path) and round_folder_name.startswith("round-"):
            for phase_folder_name in os.listdir(round_dir_path):
                phase_dir_path = os.path.join(round_dir_path, phase_folder_name)
                if os.path.isdir(phase_dir_path): # Assume que qualquer subdiretório aqui é uma fase
                    for tenant_folder_name in os.listdir(phase_dir_path):
                        tenant_dir_path = os.path.join(phase_dir_path, tenant_folder_name)
                        if os.path.isdir(tenant_dir_path): # Assume que qualquer subdiretório aqui é um tenant
                            tenants.add(tenant_folder_name)
    
    return sorted(list(tenants))


def list_available_metrics(experiment_dir, tenant="tenant-a"):
    """
    Lista todas as métricas disponíveis para um tenant específico.
    
    Args:
        experiment_dir (str): Caminho para o diretório do experimento
        tenant (str): Nome do tenant para listar métricas (pode ser None para buscar em todos)
        
    Returns:
        list: Lista de nomes de métricas
    """
    metrics = set()
    
    # Procura arquivos CSV dentro dos diretórios do tenant especificado
    # Se tenant for None, busca em qualquer diretório de tenant
    # Corrigido para iterar corretamente sobre a estrutura de diretórios
    for round_folder_name in os.listdir(experiment_dir):
        round_dir_path = os.path.join(experiment_dir, round_folder_name)
        if os.path.isdir(round_dir_path) and round_folder_name.startswith("round-"):
            for phase_folder_name in os.listdir(round_dir_path):
                phase_dir_path = os.path.join(round_dir_path, phase_folder_name)
                if os.path.isdir(phase_dir_path):
                    # Se um tenant específico for fornecido, procurar apenas nele
                    search_tenants_in_phase = [tenant] if tenant else os.listdir(phase_dir_path)
                    for tenant_folder_name in search_tenants_in_phase:
                        tenant_dir_path = os.path.join(phase_dir_path, tenant_folder_name)
                        if os.path.isdir(tenant_dir_path):
                            for file_name in os.listdir(tenant_dir_path):
                                if file_name.endswith(".csv"):
                                    metric_name = file_name.replace(".csv", "")
                                    metrics.add(metric_name)
    
    return sorted(list(metrics))


def parse_timestamp(timestamp_str):
    """
    Converte string de timestamp no formato usado nos arquivos CSV para objeto datetime.
    
    Args:
        timestamp_str (str): String de timestamp no formato "YYYYMMDD_HHMMSS"
        
    Returns:
        datetime: Objeto datetime correspondente
    """
    return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")


def load_metric_data(experiment_dir, metric_name, tenants=None, phases=None, rounds=None):
    """
    Carrega dados de uma métrica específica para os tenants, fases e rounds selecionados.
    
    Args:
        experiment_dir (str): Caminho para o diretório do experimento
        metric_name (str): Nome da métrica a ser carregada
        tenants (list): Lista de tenants a incluir (None = todos)
        phases (list): Lista de fases a incluir (None = todas)
        rounds (list): Lista de rounds a incluir (None = todos)
        
    Returns:
        DataFrame: DataFrame consolidado com os dados da métrica
    """
    all_data = []
    
    if rounds is None:
        rounds_to_scan = [r for r in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, r)) and r.startswith("round-")]
    else:
        rounds_to_scan = rounds

    if tenants is None:
        tenants_to_scan = list_available_tenants(experiment_dir)
    else:
        tenants_to_scan = tenants
    
    for round_name in rounds_to_scan:
        round_path = os.path.join(experiment_dir, round_name)
        if not os.path.isdir(round_path):
            continue

        phases_to_scan_for_round = []
        if phases is None: 
            phases_to_scan_for_round = [p for p in os.listdir(round_path) if os.path.isdir(os.path.join(round_path, p))]
        else: 
            phases_to_scan_for_round = phases

        for phase_name_pattern in phases_to_scan_for_round: 
            phase_dir_actual_path = os.path.join(round_path, phase_name_pattern)
            
            if not os.path.isdir(phase_dir_actual_path):
                continue

            phase_name_actual = os.path.basename(phase_dir_actual_path) 
            phase_number_match = re.search(r'^(\\d+)', phase_name_actual)
            phase_number = int(phase_number_match.group(1)) if phase_number_match else 0
                
            for tenant_name in tenants_to_scan:
                csv_path = os.path.join(phase_dir_actual_path, tenant_name, f"{metric_name}.csv")
                    
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        df['tenant'] = tenant_name
                        df['round'] = round_name
                        df['phase'] = phase_name_actual 
                        df['phase_number'] = phase_number
                        df['datetime'] = df['timestamp'].apply(parse_timestamp)
                        all_data.append(df)
                    except Exception as e:
                        print(f"Erro ao carregar {csv_path}: {e}")
    
    if not all_data:
        print(f"Nenhum dado encontrado para métrica '{metric_name}' com os filtros aplicados.")
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)


def load_multiple_metrics(experiment_dir, metric_names, tenants=None, phases=None, rounds=None, group_by_phase: bool = False):
    """
    Carrega dados de múltiplas métricas para os tenants, fases e rounds selecionados.
    
    Args:
        experiment_dir (str): Caminho para o diretório do experimento
        metric_names (list): Lista de nomes de métricas a serem carregadas
        tenants (list): Lista de tenants a incluir (None = todos)
        phases (list): Lista de fases a incluir (None = todas)
        rounds (list): Lista de rounds a incluir (None = todos)
        group_by_phase (bool): Se True, agrupa os dados por fase dentro de cada round.
        
    Returns:
        dict: Dicionário onde as chaves são nomes de métricas. 
              Se group_by_phase is False:
                  Cada valor é outro dicionário onde as chaves são nomes de rounds 
                  e os valores são DataFrames com os dados da métrica para aquele round.
              Se group_by_phase is True:
                  Cada valor é outro dicionário onde as chaves são nomes de rounds,
                  e cada valor de round é um dicionário onde as chaves são nomes de fases
                  e os valores são DataFrames com os dados da métrica para aquele round/fase.
    """
    metrics_data_final_structure = {}
    
    for metric in metric_names:
        df_all_rounds_for_metric = load_metric_data(experiment_dir, metric, tenants, phases, rounds)
        
        if not df_all_rounds_for_metric.empty:
            rounds_data_for_metric = {}
            if 'round' in df_all_rounds_for_metric.columns:
                for round_name, group_df_round in df_all_rounds_for_metric.groupby('round'):
                    if group_by_phase:
                        if 'phase' in group_df_round.columns:
                            phases_data_for_round = {}
                            for phase_name, group_df_phase in group_df_round.groupby('phase'):
                                # Apply intelligent unit formatting instead of hard-coded conversion
                                from ..utils.metric_formatter import detect_and_convert_units
                                
                                formatted_df = detect_and_convert_units(group_df_phase.copy(), metric)
                                phases_data_for_round[phase_name] = formatted_df
                            if phases_data_for_round:
                                rounds_data_for_metric[round_name] = phases_data_for_round
                        else:
                            print(f"Aviso: Coluna 'phase' não encontrada para a métrica {metric}, round {round_name} ao tentar agrupar por fase. Mantendo consolidado para este round.")
                            rounds_data_for_metric[round_name] = group_df_round.copy()
                    else:
                        # Apply intelligent unit formatting instead of hard-coded conversion
                        from ..utils.metric_formatter import detect_and_convert_units
                        
                        formatted_df = detect_and_convert_units(group_df_round.copy(), metric)
                        rounds_data_for_metric[round_name] = formatted_df
            else:
                print(f"Aviso: Coluna 'round' não encontrada para a métrica {metric} após carregar. Verifique a função load_metric_data.")
                if group_by_phase:
                     print(f"Aviso: Não é possível agrupar por fase para a métrica {metric} pois a coluna 'round' está ausente. Dados permanecem consolidados.")
                metrics_data_final_structure[metric] = {'unknown_round_data': df_all_rounds_for_metric}
                continue
            
            if rounds_data_for_metric: 
                metrics_data_final_structure[metric] = rounds_data_for_metric
            elif not df_all_rounds_for_metric.empty:
                 print(f"Aviso: Nenhum dado de round foi agrupado para a métrica {metric}, embora o DataFrame não estivesse vazio.")
    
    return metrics_data_final_structure


def load_experiment_data(experiment_dir, tenants=None, metrics=None, phases=None, rounds=None, group_by_phase: bool = False):
    """
    Carrega todos os dados relevantes de um experimento, potencialmente todas as métricas disponíveis.

    Args:
        experiment_dir (str): Diretório base do experimento.
        tenants (list, optional): Lista de tenants a incluir. Defaults to all available.
        metrics (list, optional): Lista de métricas a carregar. Defaults to all available for the first tenant.
        phases (list, optional): Lista de fases a incluir. Defaults to all.
        rounds (list, optional): Lista de rounds a incluir. Defaults to all.
        group_by_phase (bool): Se True, agrupa os dados por fase dentro de cada round.

    Returns:
        dict: Dicionário com DataFrames para cada métrica carregada, estruturado conforme group_by_phase.
    """
    print(f"Carregando dados do experimento de: {experiment_dir}")

    if tenants is None:
        tenants = list_available_tenants(experiment_dir)
        if not tenants:
            print("Nenhum tenant encontrado.")
            return {}
        print(f"Tenants a serem carregados: {tenants}")

    if metrics is None:
        if tenants:
            metrics = list_available_metrics(experiment_dir, tenant=tenants[0])
        if not metrics:
            print("Nenhuma métrica encontrada para os tenants especificados.")
            return {}
        print(f"Métricas a serem carregadas: {metrics}")

    experiment_data = load_multiple_metrics(
        experiment_dir,
        metric_names=metrics,
        tenants=tenants,
        phases=phases,
        rounds=rounds,
        group_by_phase=group_by_phase
    )

    if not experiment_data:
        print("Nenhum dado foi carregado para o experimento.")
    
    return experiment_data


def select_tenants(metrics_data, selected_tenants):
    """
    Filtra os dados de métricas para incluir apenas os tenants selecionados.

    Args:
        metrics_data (dict): Dicionário de DataFrames por métrica.
        selected_tenants (list): Lista de tenants para manter.

    Returns:
        dict: Dicionário de DataFrames filtrado.
    """
    if not selected_tenants:
        return metrics_data 

    filtered_data = {}
    for metric_name, df_rounds in metrics_data.items(): 
        filtered_rounds = {}
        if isinstance(df_rounds, dict): 
            for round_name, df_metric_round in df_rounds.items():
                if isinstance(df_metric_round, pd.DataFrame) and 'tenant' in df_metric_round.columns:
                    filtered_df_round = df_metric_round[df_metric_round['tenant'].isin(selected_tenants)]
                    if not filtered_df_round.empty:
                        filtered_rounds[round_name] = filtered_df_round
                else:
                    filtered_rounds[round_name] = df_metric_round 
            if filtered_rounds:
                 filtered_data[metric_name] = filtered_rounds
        elif isinstance(df_rounds, pd.DataFrame) and 'tenant' in df_rounds.columns: 
            filtered_df = df_rounds[df_rounds['tenant'].isin(selected_tenants)]
            if not filtered_df.empty:
                filtered_data[metric_name] = filtered_df
        else:
            filtered_data[metric_name] = df_rounds
            
    return filtered_data
