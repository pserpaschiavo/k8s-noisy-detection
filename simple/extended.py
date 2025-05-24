#!/usr/bin/env python3
"""
K8s Noisy Detection - Extended Analysis Module
==============================================

Módulo de análise intermediária para o sistema simplificado K8s.
Inclui técnicas avançadas como PCA, ICA, clustering e detecção de anomalias.

Autor: Phil
Versão: 2.0 (Simplificado)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Scientific computing imports
try:
    from sklearn.decomposition import PCA, FastICA
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from scipy import stats
    from scipy.signal import savgol_filter
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ scikit-learn não disponível, algumas funcionalidades serão limitadas")
    SKLEARN_AVAILABLE = False

# Add parent directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from core import SimpleK8sAnalyzer


class ExtendedK8sAnalyzer(SimpleK8sAnalyzer):
    """
    Analisador estendido com técnicas avançadas de análise.
    
    Funcionalidades adicionais:
    - Análise de componentes principais (PCA)
    - Análise de componentes independentes (ICA)  
    - Clustering (K-means, DBSCAN)
    - Detecção de anomalias
    - Análise de correlações avançadas
    - Filtragem de sinais
    """
    
    def __init__(self, config_path: Optional[str] = None, data_path: Optional[str] = None):
        """Inicializa o analisador estendido."""
        super().__init__(config_path, data_path)
        
        # Extended analysis containers
        self.scaled_data = {}
        self.pca_results = {}
        self.ica_results = {}
        self.clustering_results = {}
        self.anomaly_results = {}
        self.correlation_results = {}
        
        print(f"🔧 ExtendedK8sAnalyzer inicializado com {len(self.config['advanced']['correlation_methods'])} métodos de correlação")
    
    def extended_analysis(self) -> Dict[str, Any]:
        """
        Executa análise estendida completa (< 15 minutos).
        
        Returns:
            Dicionário com resultados da análise estendida
        """
        print("\n🔍 Iniciando análise estendida...")
        
        # 1. Análise básica primeiro
        basic_results = self.quick_analysis()
        
        if not SKLEARN_AVAILABLE:
            print("⚠️ Análise limitada sem scikit-learn")
            return basic_results
        
        # 2. Preparação de dados para análise avançada
        print("⚙️ Preparando dados para análise avançada...")
        self.prepare_advanced_data()
        
        # 3. PCA Analysis
        print("📊 Executando análise PCA...")
        self.perform_pca_analysis()
        
        # 4. ICA Analysis
        print("🔄 Executando análise ICA...")
        self.perform_ica_analysis()
        
        # 5. Clustering
        print("🎯 Executando clustering...")
        self.perform_clustering_analysis()
        
        # 6. Anomaly Detection
        print("🚨 Executando detecção de anomalias...")
        self.perform_anomaly_detection()
        
        # 7. Advanced Correlations
        print("📈 Executando análise de correlações avançadas...")
        self.perform_advanced_correlations()
        
        # 8. Create extended plots
        print("📊 Criando plots estendidos...")
        self.create_extended_plots()
        
        # 9. Generate extended report
        print("📋 Gerando relatório estendido...")
        extended_summary = self.generate_extended_report()
        
        print(f"\n✅ Análise estendida completa! Resultados em: {self.output_dir}")
        
        return {
            **basic_results,
            'pca_results': self.pca_results,
            'ica_results': self.ica_results,
            'clustering_results': self.clustering_results,
            'anomaly_results': self.anomaly_results,
            'correlation_results': self.correlation_results,
            'extended_summary': extended_summary
        }
    
    def prepare_advanced_data(self) -> None:
        """Prepara dados para análise avançada com normalização."""
        self.scaled_data = {}
        
        for phase_name, phase_data in self.processed_data.items():
            self.scaled_data[phase_name] = {}
            
            for tenant, tenant_data in phase_data.items():
                # Combina todas as métricas em uma matriz
                metric_matrix = []
                metric_names = []
                
                for metric, df in tenant_data.items():
                    if 'value' in df.columns:
                        values = df['value'].dropna()
                        if len(values) > 0:
                            metric_matrix.append(values.values)
                            metric_names.append(metric)
                
                if metric_matrix:
                    # Preenche com zeros para equalizar tamanhos
                    max_len = max(len(arr) for arr in metric_matrix)
                    padded_matrix = []
                    
                    for arr in metric_matrix:
                        if len(arr) < max_len:
                            padded = np.pad(arr, (0, max_len - len(arr)), mode='edge')
                        else:
                            padded = arr[:max_len]
                        padded_matrix.append(padded)
                    
                    # Transpõe para ter samples x features
                    data_matrix = np.array(padded_matrix).T
                    
                    # Normalização
                    if self.config['processing']['normalize']:
                        scaler = StandardScaler()
                        data_matrix = scaler.fit_transform(data_matrix)
                    
                    self.scaled_data[phase_name][tenant] = {
                        'data': data_matrix,
                        'metric_names': metric_names,
                        'sample_count': data_matrix.shape[0],
                        'feature_count': data_matrix.shape[1]
                    }
                    
                    print(f"    ✅ {tenant}: {data_matrix.shape[0]}x{data_matrix.shape[1]} matriz preparada")
    
    def perform_pca_analysis(self) -> None:
        """Executa análise de componentes principais."""
        self.pca_results = {}
        
        for phase_name, phase_data in self.scaled_data.items():
            self.pca_results[phase_name] = {}
            
            for tenant, data_info in phase_data.items():
                data_matrix = data_info['data']
                
                # Determina número de componentes
                n_components = self.config['advanced']['pca_components']
                if n_components == 'auto':
                    n_components = min(data_matrix.shape[1], 3)
                
                try:
                    pca = PCA(n_components=n_components)
                    transformed = pca.fit_transform(data_matrix)
                    
                    self.pca_results[phase_name][tenant] = {
                        'components': pca.components_,
                        'explained_variance_ratio': pca.explained_variance_ratio_,
                        'transformed_data': transformed,
                        'feature_names': data_info['metric_names'],
                        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
                    }
                    
                    variance_total = np.sum(pca.explained_variance_ratio_) * 100
                    print(f"    ✅ {tenant}: {n_components} componentes explicam {variance_total:.1f}% da variância")
                    
                except Exception as e:
                    print(f"    ❌ Erro PCA para {tenant}: {e}")
    
    def perform_ica_analysis(self) -> None:
        """Executa análise de componentes independentes."""
        self.ica_results = {}
        
        for phase_name, phase_data in self.scaled_data.items():
            self.ica_results[phase_name] = {}
            
            for tenant, data_info in phase_data.items():
                data_matrix = data_info['data']
                
                n_components = self.config['advanced']['ica_components']
                if n_components == 'auto':
                    n_components = min(data_matrix.shape[1], 3)
                
                try:
                    ica = FastICA(n_components=n_components, random_state=42)
                    transformed = ica.fit_transform(data_matrix)
                    
                    self.ica_results[phase_name][tenant] = {
                        'components': ica.components_,
                        'mixing_matrix': ica.mixing_,
                        'transformed_data': transformed,
                        'feature_names': data_info['metric_names']
                    }
                    
                    print(f"    ✅ {tenant}: {n_components} componentes independentes extraídos")
                    
                except Exception as e:
                    print(f"    ❌ Erro ICA para {tenant}: {e}")
    
    def perform_clustering_analysis(self) -> None:
        """Executa análise de clustering."""
        self.clustering_results = {}
        
        for phase_name, phase_data in self.scaled_data.items():
            self.clustering_results[phase_name] = {}
            
            for tenant, data_info in phase_data.items():
                data_matrix = data_info['data']
                self.clustering_results[phase_name][tenant] = {}
                
                # K-means clustering
                if 'kmeans' in self.config['advanced']['clustering_methods']:
                    try:
                        # Determina número ótimo de clusters (2-5)
                        best_k = 2
                        best_inertia = float('inf')
                        
                        for k in range(2, min(6, data_matrix.shape[0] // 2)):
                            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                            labels = kmeans.fit_predict(data_matrix)
                            if kmeans.inertia_ < best_inertia:
                                best_inertia = kmeans.inertia_
                                best_k = k
                        
                        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                        labels = kmeans.fit_predict(data_matrix)
                        
                        self.clustering_results[phase_name][tenant]['kmeans'] = {
                            'labels': labels,
                            'n_clusters': best_k,
                            'cluster_centers': kmeans.cluster_centers_,
                            'inertia': kmeans.inertia_
                        }
                        
                        print(f"    ✅ {tenant}: K-means com {best_k} clusters")
                        
                    except Exception as e:
                        print(f"    ❌ Erro K-means para {tenant}: {e}")
                
                # DBSCAN clustering
                if 'dbscan' in self.config['advanced']['clustering_methods']:
                    try:
                        dbscan = DBSCAN(eps=0.5, min_samples=5)
                        labels = dbscan.fit_predict(data_matrix)
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        
                        self.clustering_results[phase_name][tenant]['dbscan'] = {
                            'labels': labels,
                            'n_clusters': n_clusters,
                            'n_noise': list(labels).count(-1)
                        }
                        
                        print(f"    ✅ {tenant}: DBSCAN com {n_clusters} clusters")
                        
                    except Exception as e:
                        print(f"    ❌ Erro DBSCAN para {tenant}: {e}")
    
    def perform_anomaly_detection(self) -> None:
        """Executa detecção de anomalias."""
        self.anomaly_results = {}
        
        for phase_name, phase_data in self.scaled_data.items():
            self.anomaly_results[phase_name] = {}
            
            for tenant, data_info in phase_data.items():
                data_matrix = data_info['data']
                self.anomaly_results[phase_name][tenant] = {}
                
                # Isolation Forest
                try:
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    anomaly_labels = iso_forest.fit_predict(data_matrix)
                    anomaly_scores = iso_forest.decision_function(data_matrix)
                    
                    n_anomalies = np.sum(anomaly_labels == -1)
                    anomaly_rate = n_anomalies / len(anomaly_labels) * 100
                    
                    self.anomaly_results[phase_name][tenant]['isolation_forest'] = {
                        'labels': anomaly_labels,
                        'scores': anomaly_scores,
                        'n_anomalies': n_anomalies,
                        'anomaly_rate': anomaly_rate
                    }
                    
                    print(f"    ✅ {tenant}: {n_anomalies} anomalias detectadas ({anomaly_rate:.1f}%)")
                    
                except Exception as e:
                    print(f"    ❌ Erro Isolation Forest para {tenant}: {e}")
    
    def perform_advanced_correlations(self) -> None:
        """Executa análise de correlações avançadas."""
        self.correlation_results = {}
        
        methods = self.config['advanced']['correlation_methods']
        
        for phase_name, phase_data in self.scaled_data.items():
            self.correlation_results[phase_name] = {}
            
            for tenant, data_info in phase_data.items():
                data_matrix = data_info['data']
                feature_names = data_info['metric_names']
                
                self.correlation_results[phase_name][tenant] = {}
                
                # DataFrame para facilitar cálculos
                df = pd.DataFrame(data_matrix, columns=feature_names)
                
                for method in methods:
                    try:
                        if method == 'pearson':
                            corr_matrix = df.corr(method='pearson')
                        elif method == 'spearman':
                            corr_matrix = df.corr(method='spearman')
                        elif method == 'kendall':
                            corr_matrix = df.corr(method='kendall')
                        else:
                            continue
                        
                        self.correlation_results[phase_name][tenant][method] = corr_matrix
                        
                        # Encontra correlações mais fortes
                        abs_corr = corr_matrix.abs()
                        strong_corr = []
                        
                        for i in range(len(abs_corr.columns)):
                            for j in range(i+1, len(abs_corr.columns)):
                                corr_val = abs_corr.iloc[i, j]
                                if corr_val > 0.7:  # Correlação forte
                                    strong_corr.append({
                                        'feature1': abs_corr.columns[i],
                                        'feature2': abs_corr.columns[j],
                                        'correlation': corr_matrix.iloc[i, j]
                                    })
                        
                        print(f"    ✅ {tenant}/{method}: {len(strong_corr)} correlações fortes encontradas")
                        
                    except Exception as e:
                        print(f"    ❌ Erro correlação {method} para {tenant}: {e}")
    
    def create_extended_plots(self) -> None:
        """Cria plots estendidos."""
        plots_dir = self.output_dir / "plots_extended"
        plots_dir.mkdir(exist_ok=True)
        
        # PCA plots
        if self.pca_results:
            self._create_pca_plots(plots_dir)
        
        # ICA plots
        if self.ica_results:
            self._create_ica_plots(plots_dir)
        
        # Clustering plots
        if self.clustering_results:
            self._create_clustering_plots(plots_dir)
        
        # Anomaly plots
        if self.anomaly_results:
            self._create_anomaly_plots(plots_dir)
        
        # Advanced correlation plots
        if self.correlation_results:
            self._create_advanced_correlation_plots(plots_dir)
        
        print("✅ Plots estendidos criados")
    
    def _create_pca_plots(self, plots_dir):
        """Cria plots de análise PCA."""
        print("    📊 Criando plots PCA...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análise de Componentes Principais (PCA)', fontsize=16)
        
        # Coleta dados de todas as fases
        all_variance_ratios = []
        all_labels = []
        
        for phase_name, phase_data in self.pca_results.items():
            for tenant, pca_data in phase_data.items():
                all_variance_ratios.append(pca_data['explained_variance_ratio'])
                all_labels.append(f"{tenant}_{phase_name}")
        
        if all_variance_ratios:
            # Plot 1: Explained variance ratio
            ax = axes[0, 0]
            for i, (ratios, label) in enumerate(zip(all_variance_ratios, all_labels)):
                ax.bar(np.arange(len(ratios)) + i*0.1, ratios, width=0.1, label=label[:10], alpha=0.7)
            ax.set_title('Variância Explicada por Componente')
            ax.set_xlabel('Componente PCA')
            ax.set_ylabel('Variância Explicada')
            ax.legend()
            
            # Plot 2: Cumulative variance
            ax = axes[0, 1]
            for ratios, label in zip(all_variance_ratios, all_labels):
                cumsum = np.cumsum(ratios)
                ax.plot(cumsum, 'o-', label=label[:10], alpha=0.7)
            ax.set_title('Variância Cumulativa')
            ax.set_xlabel('Componente PCA')
            ax.set_ylabel('Variância Cumulativa')
            ax.legend()
        
        plt.tight_layout()
        pca_file = plots_dir / "pca_analysis.png"
        plt.savefig(pca_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✅ Salvo: {pca_file}")
    
    def _create_clustering_plots(self, plots_dir):
        """Cria plots de clustering."""
        print("    📊 Criando plots de clustering...")
        
        for phase_name, phase_data in self.clustering_results.items():
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Análise de Clustering - {phase_name}', fontsize=16)
            
            tenant_idx = 0
            for tenant, clustering_data in phase_data.items():
                if tenant_idx >= 4:
                    break
                
                row = tenant_idx // 2
                col = tenant_idx % 2
                ax = axes[row, col]
                
                # Plot K-means se disponível
                if 'kmeans' in clustering_data:
                    kmeans_data = clustering_data['kmeans']
                    labels = kmeans_data['labels']
                    
                    # Plot dos clusters (usando as duas primeiras dimensões dos dados)
                    data_matrix = self.scaled_data[phase_name][tenant]['data']
                    if data_matrix.shape[1] >= 2:
                        scatter = ax.scatter(data_matrix[:, 0], data_matrix[:, 1], 
                                           c=labels, cmap='viridis', alpha=0.6)
                        ax.set_title(f'{tenant} - K-means ({kmeans_data["n_clusters"]} clusters)')
                        ax.set_xlabel('Feature 1')
                        ax.set_ylabel('Feature 2')
                        plt.colorbar(scatter, ax=ax)
                
                tenant_idx += 1
            
            plt.tight_layout()
            cluster_file = plots_dir / f"clustering_{phase_name.replace(' ', '_')}.png"
            plt.savefig(cluster_file, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"      ✅ Plots de clustering salvos")
    
    def _create_anomaly_plots(self, plots_dir):
        """Cria plots de detecção de anomalias."""
        print("    📊 Criando plots de anomalias...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Detecção de Anomalias', fontsize=16)
        
        phase_idx = 0
        for phase_name, phase_data in self.anomaly_results.items():
            if phase_idx >= 4:
                break
                
            ax = axes[phase_idx // 2, phase_idx % 2]
            
            # Coleta dados de anomalias de todos os tenants desta fase
            anomaly_rates = []
            tenant_names = []
            
            for tenant, anomaly_data in phase_data.items():
                if 'isolation_forest' in anomaly_data:
                    rate = anomaly_data['isolation_forest']['anomaly_rate']
                    anomaly_rates.append(rate)
                    tenant_names.append(tenant.replace('tenant-', ''))
            
            if anomaly_rates:
                bars = ax.bar(tenant_names, anomaly_rates, alpha=0.7, color='red')
                ax.set_title(f'{phase_name} - Taxa de Anomalias')
                ax.set_xlabel('Tenants')
                ax.set_ylabel('Taxa de Anomalias (%)')
                ax.set_ylim(0, max(anomaly_rates) * 1.2)
                
                # Adiciona valores nas barras
                for bar, rate in zip(bars, anomaly_rates):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{rate:.1f}%', ha='center', va='bottom')
            
            phase_idx += 1
        
        plt.tight_layout()
        anomaly_file = plots_dir / "anomaly_detection.png"
        plt.savefig(anomaly_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✅ Salvo: {anomaly_file}")
    
    def _create_advanced_correlation_plots(self, plots_dir):
        """Cria plots de correlações avançadas."""
        print("    📊 Criando plots de correlações avançadas...")
        
        methods = self.config['advanced']['correlation_methods']
        
        for method in methods[:3]:  # Limite para os 3 primeiros métodos
            if method not in ['pearson', 'spearman', 'kendall']:
                continue
                
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Correlações {method.capitalize()}', fontsize=16)
            
            phase_idx = 0
            for phase_name, phase_data in self.correlation_results.items():
                if phase_idx >= 3:
                    break
                    
                ax = axes[phase_idx]
                
                # Média das correlações de todos os tenants
                all_corr_matrices = []
                
                for tenant, corr_data in phase_data.items():
                    if method in corr_data:
                        all_corr_matrices.append(corr_data[method].values)
                
                if all_corr_matrices:
                    mean_corr = np.mean(all_corr_matrices, axis=0)
                    feature_names = list(phase_data[list(phase_data.keys())[0]][method].columns)
                    
                    im = ax.imshow(mean_corr, cmap='coolwarm', vmin=-1, vmax=1)
                    ax.set_title(f'{phase_name}')
                    ax.set_xticks(range(len(feature_names)))
                    ax.set_yticks(range(len(feature_names)))
                    ax.set_xticklabels([name.replace('_', '\n') for name in feature_names], rotation=45)
                    ax.set_yticklabels([name.replace('_', '\n') for name in feature_names])
                    
                    # Adiciona valores nas células
                    for i in range(len(feature_names)):
                        for j in range(len(feature_names)):
                            text = ax.text(j, i, f'{mean_corr[i, j]:.2f}',
                                         ha="center", va="center", color="black", fontsize=8)
                
                phase_idx += 1
            
            plt.tight_layout()
            corr_file = plots_dir / f"correlation_{method}.png"
            plt.savefig(corr_file, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"      ✅ Plots de correlação salvos")
    
    def _create_ica_plots(self, plots_dir):
        """Cria plots de análise ICA."""
        print("    📊 Criando plots ICA...")
        
        # ICA é mais complexo de visualizar, vamos fazer um plot simples dos componentes
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Análise de Componentes Independentes (ICA)', fontsize=16)
        
        phase_idx = 0
        for phase_name, phase_data in self.ica_results.items():
            if phase_idx >= 4:
                break
                
            ax = axes[phase_idx // 2, phase_idx % 2]
            
            # Visualiza mixing matrix de todos os tenants
            for tenant_idx, (tenant, ica_data) in enumerate(phase_data.items()):
                mixing_matrix = ica_data['mixing_matrix']
                
                # Plot das primeiras duas colunas da mixing matrix
                if mixing_matrix.shape[1] >= 2:
                    ax.scatter(mixing_matrix[:, 0], mixing_matrix[:, 1], 
                             label=tenant.replace('tenant-', ''), alpha=0.7)
            
            ax.set_title(f'{phase_name} - Mixing Matrix')
            ax.set_xlabel('Componente ICA 1')
            ax.set_ylabel('Componente ICA 2')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            phase_idx += 1
        
        plt.tight_layout()
        ica_file = plots_dir / "ica_analysis.png"
        plt.savefig(ica_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✅ Salvo: {ica_file}")
    
    def generate_extended_report(self) -> Dict[str, Any]:
        """Gera relatório estendido."""
        print("\n📄 GERANDO RELATÓRIO ESTENDIDO...")
        
        # Relatório texto estendido
        report_file = self.output_dir / "extended_report.txt"
        with open(report_file, 'w') as f:
            f.write("RELATÓRIO ESTENDIDO - ANÁLISE AVANÇADA K8S\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Gerado em: {datetime.now()}\n")
            f.write(f"Fonte dos dados: {self.data_path}\n")
            f.write(f"Técnicas aplicadas: {self.config['analysis']['techniques']}\n\n")
            
            # Seção PCA
            if self.pca_results:
                f.write("ANÁLISE PCA:\n")
                f.write("-" * 30 + "\n")
                for phase_name, phase_data in self.pca_results.items():
                    f.write(f"\n{phase_name}:\n")
                    for tenant, pca_data in phase_data.items():
                        variance_total = np.sum(pca_data['explained_variance_ratio']) * 100
                        f.write(f"  {tenant}: {len(pca_data['explained_variance_ratio'])} componentes, ")
                        f.write(f"{variance_total:.1f}% variância explicada\n")
                f.write("\n")
            
            # Seção Clustering
            if self.clustering_results:
                f.write("ANÁLISE DE CLUSTERING:\n")
                f.write("-" * 30 + "\n")
                for phase_name, phase_data in self.clustering_results.items():
                    f.write(f"\n{phase_name}:\n")
                    for tenant, cluster_data in phase_data.items():
                        for method, results in cluster_data.items():
                            f.write(f"  {tenant}/{method}: {results.get('n_clusters', 0)} clusters\n")
                f.write("\n")
            
            # Seção Anomalias
            if self.anomaly_results:
                f.write("DETECÇÃO DE ANOMALIAS:\n")
                f.write("-" * 30 + "\n")
                for phase_name, phase_data in self.anomaly_results.items():
                    f.write(f"\n{phase_name}:\n")
                    for tenant, anomaly_data in phase_data.items():
                        if 'isolation_forest' in anomaly_data:
                            rate = anomaly_data['isolation_forest']['anomaly_rate']
                            count = anomaly_data['isolation_forest']['n_anomalies']
                            f.write(f"  {tenant}: {count} anomalias ({rate:.1f}%)\n")
                f.write("\n")
        
        summary_data = {
            'pca_phases': len(self.pca_results),
            'clustering_methods': len(self.config['advanced']['clustering_methods']),
            'correlation_methods': len(self.config['advanced']['correlation_methods']),
            'anomaly_detection_enabled': True,
            'extended_analysis_complete': True
        }
        
        print(f"✅ Relatório estendido salvo: {report_file}")
        
        return summary_data


if __name__ == "__main__":
    # Teste do analisador estendido
    print("🚀 Testando ExtendedK8sAnalyzer...")
    
    analyzer = ExtendedK8sAnalyzer(data_path='demo-data/demo-experiment-1-round/round-1')
    result = analyzer.extended_analysis()
    
    if result:
        print("\n✅ Teste estendido concluído com sucesso!")
        print(f"📂 Resultados em: {result['output_dir']}")
    else:
        print("\n❌ Teste estendido falhou")
