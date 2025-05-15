#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase Analysis Module for Kubernetes Noisy Neighbours Lab
Este módulo implementa análises comparativas entre diferentes fases do experimento (baseline, ataque, recuperação).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import scipy.stats as stats
from analysis_pipeline.metrics_analysis import MetricsAnalyzer

class PhaseAnalyzer:
    def __init__(self, output_dir=None):
        """
        Inicializa o analisador de fases.
        
        Args:
            output_dir (str): Diretório para salvar resultados e gráficos
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir and not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # Criar subdiretórios para análises específicas
        if self.output_dir:
            self.plots_dir = self.output_dir / "phase_analysis"
            self.plots_dir.mkdir(exist_ok=True)
            self.stats_dir = self.output_dir / "phase_stats"
            self.stats_dir.mkdir(exist_ok=True)
            self.tables_dir = self.output_dir / "phase_tables"
            self.tables_dir.mkdir(exist_ok=True)
            
        # Inicializar o analisador de métricas para uso nos métodos
        self.metrics_analyzer = MetricsAnalyzer(output_dir)
            
        logging.info(f"Inicializado PhaseAnalyzer, diretório de saída: {self.output_dir}")
    
    # ========== COMPARAÇÃO ENTRE FASES ==========
    
    def compare_phases(self, data_dict, metric_name=None, phase_names=None, comparison_methods=None):
        """
        Compara métricas entre diferentes fases do experimento.
        
        Args:
            data_dict (dict): Dicionário com dados por fase {fase1: series1, fase2: series2, ...}
            metric_name (str): Nome da métrica para contexto
            phase_names (list): Nomes das fases para usar em legendas
            comparison_methods (list): Métodos de comparação a utilizar
            
        Returns:
            dict: Resultados das comparações entre fases
            
        Outputs:
            - Gráficos de comparação (.png)
            - Tabelas de resultados (.csv, .tex)
        """
        if not data_dict or len(data_dict) < 2:
            logging.warning("Dados insuficientes para comparação entre fases")
            return None
            
        # Definir métodos de comparação padrão se não especificados
        if comparison_methods is None:
            comparison_methods = ['boxplot', 'violin', 'stats_test']
            
        # Criar dicionário para armazenar resultados
        results = {}
        
        # Usar nomes das fases do dicionário se não fornecido
        if phase_names is None:
            phase_names = list(data_dict.keys())
        
        # Preparar dados
        clean_data = {}
        for phase, data in data_dict.items():
            if isinstance(data, pd.DataFrame):
                # Se for DataFrame, usar a primeira coluna numérica
                numeric_cols = data.select_dtypes(include=np.number).columns
                if len(numeric_cols) > 0:
                    clean_data[phase] = data[numeric_cols[0]]
            else:
                clean_data[phase] = pd.to_numeric(data, errors='coerce')
        
        # Verificar se temos dados suficientes após limpeza
        if len(clean_data) < 2:
            logging.warning("Dados insuficientes para comparação após limpeza")
            return None
            
        # Executar comparações solicitadas
        for method in comparison_methods:
            if method == 'boxplot':
                self._generate_boxplot_comparison(clean_data, metric_name, phase_names)
            elif method == 'violin':
                self._generate_violin_comparison(clean_data, metric_name, phase_names)
            elif method == 'stats_test':
                results['stats_test'] = self._run_statistical_tests(clean_data, metric_name, phase_names)
            elif method == 'time_series':
                self._generate_time_series_comparison(data_dict, metric_name, phase_names)
        
        return results
    
    def _generate_boxplot_comparison(self, data_dict, metric_name=None, phase_names=None):
        """Gera gráfico de boxplot para comparação entre fases"""
        if not self.plots_dir:
            return
        
        plt.figure(figsize=(12, 6))
        
        # Preparar dados para o boxplot
        plot_data = []
        plot_labels = []
        
        for phase, data in data_dict.items():
            plot_data.append(data.dropna())
            plot_labels.append(phase if phase_names is None else 
                              phase_names[list(data_dict.keys()).index(phase)])
        
        # Gerar boxplot
        plt.boxplot(plot_data, labels=plot_labels, patch_artist=True)
        
        if metric_name:
            plt.title(f"Comparação de {metric_name} entre Fases")
            plt.ylabel(metric_name)
        else:
            plt.title("Comparação entre Fases")
        
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Salvar figura
        safe_name = f"{metric_name}_phase_boxplot".replace(" ", "_").replace("/", "_") if metric_name else "phase_boxplot"
        fig_path = self.plots_dir / f"{safe_name}.png"
        plt.savefig(fig_path)
        plt.close()
        logging.info(f"Boxplot de comparação entre fases salvo em: {fig_path}")
    
    def _generate_violin_comparison(self, data_dict, metric_name=None, phase_names=None):
        """Gera gráfico de violino para comparação entre fases"""
        if not self.plots_dir:
            return
        
        plt.figure(figsize=(12, 6))
        
        # Preparar dados para o violino
        df_list = []
        for phase, data in data_dict.items():
            phase_name = phase if phase_names is None else phase_names[list(data_dict.keys()).index(phase)]
            temp_df = pd.DataFrame({
                'value': data.dropna(),
                'phase': phase_name
            })
            df_list.append(temp_df)
        
        if not df_list:
            logging.warning("Dados insuficientes para gráfico de violino")
            return
            
        plot_df = pd.concat(df_list)
        
        # Gerar violino
        sns.violinplot(data=plot_df, x='phase', y='value')
        
        if metric_name:
            plt.title(f"Distribuição de {metric_name} por Fase")
            plt.ylabel(metric_name)
        else:
            plt.title("Distribuição por Fase")
        
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Salvar figura
        safe_name = f"{metric_name}_phase_violin".replace(" ", "_").replace("/", "_") if metric_name else "phase_violin"
        fig_path = self.plots_dir / f"{safe_name}.png"
        plt.savefig(fig_path)
        plt.close()
        logging.info(f"Violin plot de comparação entre fases salvo em: {fig_path}")
    
    def _generate_time_series_comparison(self, data_dict, metric_name=None, phase_names=None):
        """Gera gráfico de série temporal com dados sobrepostos de diferentes fases"""
        if not self.plots_dir:
            return
        
        plt.figure(figsize=(14, 6))
        
        # Preparar dados e legendas
        for i, (phase, data) in enumerate(data_dict.items()):
            # Determinar rótulo para a legenda
            label = phase if phase_names is None else phase_names[list(data_dict.keys()).index(phase)]
            
            # Se o dado for DataFrame, extrair a primeira série
            if isinstance(data, pd.DataFrame):
                numeric_cols = data.select_dtypes(include=np.number).columns
                if len(numeric_cols) > 0:
                    # Usar diferentes estilos de linha para cada fase
                    plt.plot(data[numeric_cols[0]], label=label, 
                             linestyle=['-', '--', '-.', ':'][i % 4],
                             linewidth=2)
            else:
                # Converter para numérico se necessário
                series = pd.to_numeric(data, errors='coerce')
                plt.plot(series, label=label,
                         linestyle=['-', '--', '-.', ':'][i % 4],
                         linewidth=2)
        
        if metric_name:
            plt.title(f"Comparação de {metric_name} ao Longo das Fases")
            plt.ylabel(metric_name)
        else:
            plt.title("Comparação de Séries Temporais entre Fases")
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Adicionar sombreamento entre fases se possível
        # [código adicional aqui se informações de limite de fase estiverem disponíveis]
        
        # Salvar figura
        safe_name = f"{metric_name}_phase_timeseries".replace(" ", "_").replace("/", "_") if metric_name else "phase_timeseries"
        fig_path = self.plots_dir / f"{safe_name}.png"
        plt.savefig(fig_path)
        plt.close()
        logging.info(f"Gráfico de série temporal comparativa salvo em: {fig_path}")
    
    def _run_statistical_tests(self, data_dict, metric_name=None, phase_names=None):
        """Executa testes estatísticos comparando dados entre fases"""
        results = {}
        
        if len(data_dict) < 2:
            return {"error": "Precisa de pelo menos duas fases para comparação"}
            
        # Considerar todas as combinações de pares
        phases = list(data_dict.keys())
        
        for i, phase1 in enumerate(phases):
            for phase2 in phases[i+1:]:
                data1 = data_dict[phase1].dropna()
                data2 = data_dict[phase2].dropna()
                
                if len(data1) < 5 or len(data2) < 5:
                    continue
                    
                # Nomes para reportar
                name1 = phase1 if phase_names is None else phase_names[phases.index(phase1)]
                name2 = phase2 if phase_names is None else phase_names[phases.index(phase2)]
                comparison_name = f"{name1} vs {name2}"
                
                # Executar teste-t
                try:
                    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                    significant = p_value < 0.05
                    
                    # Calcular tamanho do efeito (Cohen's d)
                    mean1, mean2 = data1.mean(), data2.mean()
                    pooled_std = np.sqrt((data1.std()**2 + data2.std()**2) / 2)
                    effect_size = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                    
                    # Calcular mudança percentual
                    if mean1 != 0:
                        percent_change = ((mean2 - mean1) / abs(mean1)) * 100
                    else:
                        percent_change = np.nan
                        
                    results[comparison_name] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': significant,
                        'effect_size': effect_size,
                        'mean1': mean1,
                        'mean2': mean2,
                        'percent_change': percent_change
                    }
                except Exception as e:
                    logging.warning(f"Erro ao executar teste estatístico: {str(e)}")
        
        # Salvar resultados como tabela
        if results and self.tables_dir and metric_name:
            # Converter para DataFrame para facilitar exportação
            result_rows = []
            for comparison, stats_result in results.items():
                result_rows.append({
                    'Comparison': comparison,
                    'T-statistic': stats_result.get('t_statistic'),
                    'p-value': stats_result.get('p_value'),
                    'Significant': stats_result.get('significant'),
                    'Effect Size': stats_result.get('effect_size'),
                    'Mean (1st)': stats_result.get('mean1'),
                    'Mean (2nd)': stats_result.get('mean2'),
                    'Percent Change': stats_result.get('percent_change')
                })
                
            result_df = pd.DataFrame(result_rows)
            
            # Salvar em CSV e LaTeX
            safe_name = f"{metric_name}_phase_comparison".replace(" ", "_").replace("/", "_")
            csv_path = self.stats_dir / f"{safe_name}_stats.csv"
            result_df.to_csv(csv_path, index=False)
            logging.info(f"Resultados estatísticos salvos em: {csv_path}")
            
            # Gerar tabela LaTeX
            tex_path = self.stats_dir / f"{safe_name}_stats.tex"
            with open(tex_path, 'w') as f:
                f.write(result_df.to_latex(index=False, float_format=lambda x: f"{x:.3f}"))
            logging.info(f"Tabela LaTeX salva em: {tex_path}")
        
        return results
    
    # ========== DETECÇÃO DE PONTOS DE MUDANÇA ==========
    
    def detect_change_points(self, data, metric_name=None, method='rpt', penalty='bic', min_size=10):
        """
        Detecta pontos de mudança significativa em uma série temporal.
        
        Args:
            data (Series): Série temporal para análise
            metric_name (str): Nome da métrica para contexto
            method (str): Método de detecção ('rpt', 'window', 'cusum')
            penalty (str): Penalidade para métodos baseados em custo ('bic', 'mbic', 'aic')
            min_size (int): Tamanho mínimo entre mudanças
            
        Returns:
            list: Índices dos pontos de mudança detectados
            
        Outputs:
            - Gráfico com pontos de mudança marcados (.png)
        """
        try:
            import ruptures as rpt
        except ImportError:
            logging.warning("Biblioteca 'ruptures' não encontrada. Instalando via pip...")
            try:
                import pip
                pip.main(['install', 'ruptures'])
                import ruptures as rpt
            except Exception as e:
                logging.error(f"Não foi possível instalar ou importar 'ruptures': {str(e)}")
                return []
                
        # Converter para numérico e remover NaNs
        clean_data = pd.to_numeric(data, errors='coerce').dropna()
        
        if len(clean_data) < min_size * 2:
            logging.warning("Dados insuficientes para detecção de pontos de mudança")
            return []
            
        # Converter para array numpy para usar com ruptures
        signal = clean_data.values
        
        # Selecionar algoritmo de detecção
        if method == 'rpt':
            algo = rpt.Pelt(model="rbf", min_size=min_size).fit(signal)
            change_points = algo.predict(pen=penalty)
        elif method == 'window':
            algo = rpt.Window(width=40, model="rbf").fit(signal)
            change_points = algo.predict(n_bkps=5)  # Detectar até 5 pontos
        elif method == 'cusum':
            # CUSUM é mais sensível a mudanças em incrementos
            diff_signal = np.diff(signal)
            algo = rpt.KernelCPD(kernel="linear", min_size=min_size).fit(np.column_stack((signal[:-1], diff_signal)))
            change_points = algo.predict(pen=penalty)
        else:
            logging.error(f"Método de detecção desconhecido: {method}")
            return []
            
        # Se for vazio, tentar com menos penalidade
        if not change_points and method == 'rpt':
            algo = rpt.Pelt(model="rbf", min_size=min_size//2).fit(signal)
            change_points = algo.predict(pen=penalty.lower())
            
        # Converter para índices da série original
        cp_indices = [clean_data.index[cp-1] if cp > 0 else clean_data.index[0] for cp in change_points if cp < len(clean_data)]
        
        # Gerar visualização
        if self.plots_dir and metric_name:
            plt.figure(figsize=(12, 6))
            plt.plot(clean_data, label='Série Original')
            
            # Marcar pontos de mudança
            for cp in cp_indices:
                plt.axvline(x=cp, color='red', linestyle='--')
                
            plt.title(f"Detecção de Pontos de Mudança: {metric_name}")
            plt.xlabel("Tempo")
            plt.ylabel(metric_name)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(['Dados', 'Pontos de Mudança'])
            
            # Salvar figura
            safe_name = f"{metric_name}_change_points".replace(" ", "_").replace("/", "_")
            fig_path = self.plots_dir / f"{safe_name}.png"
            plt.savefig(fig_path)
            plt.close()
            logging.info(f"Gráfico de pontos de mudança salvo em: {fig_path}")
            
        return cp_indices
    
    # ========== ANÁLISE DE RECUPERAÇÃO ==========
    
    def analyze_recovery(self, phases_data, baseline_phase, attack_phase, recovery_phase, metric_name=None):
        """
        Analisa o processo de recuperação após um ataque.
        
        Args:
            phases_data (dict): Dicionário com dados por fase
            baseline_phase (str): Nome da fase de linha de base
            attack_phase (str): Nome da fase de ataque
            recovery_phase (str): Nome da fase de recuperação
            metric_name (str): Nome da métrica para contexto
            
        Returns:
            dict: Resultados da análise de recuperação
            
        Outputs:
            - Gráfico de recuperação (.png)
            - Tabela com métricas de recuperação (.csv, .tex)
        """
        if not all(phase in phases_data for phase in [baseline_phase, attack_phase, recovery_phase]):
            logging.warning("Faltam dados de fase necessários para análise de recuperação")
            return None
            
        # Extrair séries
        baseline = phases_data[baseline_phase]
        attack = phases_data[attack_phase]
        recovery = phases_data[recovery_phase]
        
        # Calcular estatísticas de referência
        baseline_mean = baseline.mean()
        baseline_std = baseline.std()
        
        attack_mean = attack.mean()
        attack_std = attack.std()
        
        # Calcular o desvio do ataque em relação ao baseline
        attack_deviation = attack_mean - baseline_mean
        attack_deviation_percent = (attack_deviation / baseline_mean) * 100 if baseline_mean != 0 else float('inf')
        
        # Analisar recuperação
        recovery_stats = {}
        recovery_stats['baseline_mean'] = baseline_mean
        recovery_stats['attack_mean'] = attack_mean
        recovery_stats['attack_deviation'] = attack_deviation
        recovery_stats['attack_deviation_percent'] = attack_deviation_percent
        
        # Verificar tendência de recuperação
        recovery_trend = None
        if len(recovery) >= 10:  # Precisa de dados suficientes
            from scipy import stats as sp_stats
            
            # Calcular coeficiente de correlação entre tempo e valor para detectar tendência
            time_indices = np.arange(len(recovery))
            corr, p_value = sp_stats.pearsonr(time_indices, recovery.values)
            
            if p_value < 0.05:  # Correlação significativa
                recovery_trend = "increasing" if corr > 0 else "decreasing"
            else:
                recovery_trend = "stable"
                
        recovery_stats['recovery_trend'] = recovery_trend
        
        # Estimar tempo para recuperação completa
        recovery_complete = False
        recovery_percent = 0
        recovery_time_estimate = None
        
        # Verificar se os valores de recuperação estão voltando ao normal
        if attack_deviation != 0:
            # Calcular quanto já recuperou (em percentual)
            recovery_deviation = recovery.iloc[-1] - attack_mean
            recovery_direction = -1 if attack_deviation > 0 else 1  # Direção esperada da recuperação
            
            # Verificar se a tendência de recuperação está na direção esperada
            if recovery_direction * recovery_deviation > 0:
                recovery_percent = abs(recovery_deviation / attack_deviation) * 100
                recovery_stats['recovery_percent'] = min(recovery_percent, 100)
                
                # Verificar se já está completamente recuperado
                if abs(recovery.iloc[-1] - baseline_mean) <= baseline_std:
                    recovery_complete = True
                
                # Se não recuperou completamente, estimar tempo restante
                elif recovery_trend in ["increasing", "decreasing"] and len(recovery) >= 5:
                    # Calcular taxa de mudança média por ponto
                    diffs = recovery.diff().dropna()
                    rate_of_change = diffs.mean()
                    
                    if rate_of_change * recovery_direction > 0:  # Taxa de mudança na direção esperada
                        remaining_change = abs(recovery.iloc[-1] - baseline_mean)
                        time_est = remaining_change / abs(rate_of_change)
                        recovery_time_estimate = time_est
            else:
                # Não está recuperando na direção esperada
                recovery_percent = 0
        
        recovery_stats['recovery_complete'] = recovery_complete
        recovery_stats['recovery_time_estimate'] = recovery_time_estimate
        
        # Gerar visualização
        if self.plots_dir and metric_name:
            plt.figure(figsize=(14, 8))
            
            # Plotar as três fases com diferentes estilos
            plt.plot(baseline, color='green', label=f'Baseline (média: {baseline_mean:.2f})', alpha=0.8)
            plt.plot(attack, color='red', label=f'Ataque (média: {attack_mean:.2f})', alpha=0.8)
            plt.plot(recovery, color='blue', label=f'Recuperação' + 
                     (f' ({recovery_percent:.1f}% recuperado)' if recovery_percent > 0 else ''), 
                     alpha=0.8)
            
            # Plotar linhas horizontais de referência
            plt.axhline(y=baseline_mean, color='green', linestyle='--', alpha=0.5)
            plt.axhline(y=attack_mean, color='red', linestyle='--', alpha=0.5)
            
            # Adicionar projeção de recuperação se houver estimativa
            if recovery_time_estimate and not recovery_complete:
                last_rec_point = recovery.iloc[-1]
                last_rec_time = recovery.index[-1]
                est_full_recovery_time = last_rec_time + pd.Timedelta(seconds=recovery_time_estimate)
                plt.plot([last_rec_time, est_full_recovery_time], 
                       [last_rec_point, baseline_mean], 'b--', alpha=0.5)
                plt.scatter(est_full_recovery_time, baseline_mean, marker='X', color='blue', 
                          s=100, label=f'Recuperação Estimada')
            
            plt.title(f"Análise de Recuperação: {metric_name}")
            plt.ylabel(metric_name)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # Destacar regiões
            min_val = min(baseline.min(), attack.min(), recovery.min())
            max_val = max(baseline.max(), attack.max(), recovery.max())
            y_range = max_val - min_val
            
            # Adicionar anotações
            if recovery_complete:
                plt.annotate('Completamente Recuperado', 
                           xy=(recovery.index[-1], recovery.iloc[-1]),
                           xytext=(recovery.index[-1], recovery.iloc[-1] + y_range * 0.1),
                           arrowprops=dict(facecolor='green', shrink=0.05))
            elif recovery_percent > 0:
                plt.annotate(f'{recovery_percent:.1f}% Recuperado', 
                           xy=(recovery.index[-1], recovery.iloc[-1]),
                           xytext=(recovery.index[-1], recovery.iloc[-1] + y_range * 0.1),
                           arrowprops=dict(facecolor='blue', shrink=0.05))
            
            # Salvar figura
            safe_name = f"{metric_name}_recovery_analysis".replace(" ", "_").replace("/", "_")
            fig_path = self.plots_dir / f"{safe_name}.png"
            plt.savefig(fig_path)
            plt.close()
            logging.info(f"Gráfico de análise de recuperação salvo em: {fig_path}")
            
            # Salvar tabela de resultados
            if self.tables_dir:
                # Criar DataFrame com resultados
                result_df = pd.DataFrame({
                    'Métrica': [
                        'Valor Médio Baseline', 
                        'Valor Médio Ataque', 
                        'Desvio do Ataque', 
                        'Desvio do Ataque (%)', 
                        'Tendência de Recuperação', 
                        'Recuperação Completa', 
                        'Percentual Recuperado',
                        'Tempo Estimado para Recuperação (s)'
                    ],
                    'Valor': [
                        f"{baseline_mean:.3f}",
                        f"{attack_mean:.3f}",
                        f"{attack_deviation:.3f}",
                        f"{attack_deviation_percent:.2f}%",
                        recovery_trend or "N/A",
                        "Sim" if recovery_complete else "Não",
                        f"{recovery_percent:.2f}%" if recovery_percent > 0 else "N/A",
                        f"{recovery_time_estimate:.1f}" if recovery_time_estimate else "N/A"
                    ]
                })
                
                # Salvar em CSV
                csv_path = self.tables_dir / f"{safe_name}_stats.csv"
                result_df.to_csv(csv_path, index=False)
                logging.info(f"Tabela de análise de recuperação salva em: {csv_path}")
                
                # Gerar tabela LaTeX
                tex_path = self.tables_dir / f"{safe_name}_stats.tex"
                with open(tex_path, 'w') as f:
                    f.write(result_df.to_latex(index=False))
                logging.info(f"Tabela LaTeX de análise de recuperação salva em: {tex_path}")
        
        return recovery_stats
    
    # ========== SISTEMA DE SUGESTÕES ==========
    
    def suggest_analyses(self, phases_data, metric_name=None):
        """
        Sugere análises relevantes com base nas características das fases.
        
        Args:
            phases_data (dict): Dicionário com dados por fase
            metric_name (str): Nome da métrica para contexto
            
        Returns:
            dict: Sugestões de análises com justificativas
        """
        suggestions = {}
        
        # Verificar número de fases
        if not phases_data or len(phases_data) < 2:
            return {"error": "Dados insuficientes para sugestões de análise de fase"}
            
        phase_names = list(phases_data.keys())
        
        # Verificar se há diferenças significativas entre fases
        try:
            # Calcular médias das fases
            phase_means = {phase: data.mean() for phase, data in phases_data.items()}
            
            # Calcular desvios padrão
            phase_stds = {phase: data.std() for phase, data in phases_data.items()}
            
            # Identificar fases com diferenças significativas
            significant_diffs = []
            
            for i, phase1 in enumerate(phase_names):
                for phase2 in phase_names[i+1:]:
                    # Calcular estatística z para diferença entre médias
                    mean_diff = abs(phase_means[phase1] - phase_means[phase2])
                    pooled_std = np.sqrt((phase_stds[phase1]**2 + phase_stds[phase2]**2) / 2)
                    
                    if pooled_std > 0:
                        z_stat = mean_diff / pooled_std
                        
                        if z_stat > 1.96:  # Diferença significativa (p < 0.05)
                            significant_diffs.append((phase1, phase2, z_stat))
            
            # Sugerir análises com base nas diferenças
            if significant_diffs:
                # Ordenar diferenças por magnitude
                significant_diffs.sort(key=lambda x: x[2], reverse=True)
                
                # Recomendar comparações entre fases específicas
                for phase1, phase2, z_stat in significant_diffs[:3]:  # Top 3 diferenças
                    comparison_name = f"{phase1}_vs_{phase2}".replace(" ", "_").replace("/", "_")
                    suggestions[comparison_name] = {
                        "type": f"Comparação Detalhada entre {phase1} e {phase2}",
                        "justification": f"Diferença significativa detectada (z={z_stat:.2f})",
                        "methods": ["boxplot", "violin", "stats_test"]
                    }
                
                # Sugerir visualização de série temporal se houver múltiplas fases
                if len(phase_names) >= 3:
                    suggestions["time_series_viz"] = {
                        "type": "Visualização de Séries Temporais Alinhadas",
                        "justification": "Múltiplas fases com diferenças significativas encontradas",
                        "methods": ["time_series"]
                    }
                    
                # Se há diferenças grandes, sugerir análise de pontos de mudança
                if any(z > 3.0 for _, _, z in significant_diffs):
                    suggestions["change_points"] = {
                        "type": "Detecção de Pontos de Mudança",
                        "justification": "Diferenças substanciais entre fases sugerem pontos de transição importantes",
                        "methods": ["detect_change_points"]
                    }
        except Exception as e:
            logging.warning(f"Erro ao sugerir análises de fase: {str(e)}")
            
        # Verificar se há fases de baseline, ataque e recuperação para análise de recuperação
        baseline_phases = [p for p in phase_names if "baseline" in p.lower()]
        attack_phases = [p for p in phase_names if "attack" in p.lower() or "ataque" in p.lower()]
        recovery_phases = [p for p in phase_names if "recovery" in p.lower() or "recuperação" in p.lower()]
        
        if baseline_phases and attack_phases and recovery_phases:
            suggestions["recovery_analysis"] = {
                "type": "Análise de Recuperação",
                "justification": "Dados de baseline, ataque e recuperação disponíveis",
                "methods": ["analyze_recovery"],
                "parameters": {
                    "baseline_phase": baseline_phases[0],
                    "attack_phase": attack_phases[0],
                    "recovery_phase": recovery_phases[0]
                }
            }
            
        return suggestions
