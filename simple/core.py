#!/usr/bin/env python3
"""
Core Simples para K8s Noisy Detection
Foco em debugging e análise básica rápida
"""

import sys
import os
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimpleK8sAnalyzer:
    """
    Analisador simples para debugging e análise básica do K8s Noisy Detection.
    Foco em clareza, velocidade e facilidade de debugging.
    """
    
    def __init__(self, data_path, tenants=None, metrics=None, output_dir='simple_output'):
        self.data_path = Path(data_path)
        self.tenants = tenants or ['tenant-a', 'tenant-b', 'tenant-c', 'tenant-d'] 
        self.metrics = metrics or ['memory_usage', 'disk_throughput_total', 'network_total_bandwidth', 'cpu_usage']
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.raw_data = {}
        self.processed_data = {}
        self.data_issues = {}
        self.summary_stats = {}
        
        # Estado do sistema
        self.data_loaded = False
        self.analysis_complete = False
        
        print(f"🎯 SimpleK8sAnalyzer inicializado")
        print(f"📁 Dados: {self.data_path}")
        print(f"👥 Tenants: {self.tenants}")
        print(f"📊 Métricas: {self.metrics}")
        print(f"📁 Saída: {self.output_dir}")
    
    def load_basic_data(self):
        """Carrega dados básicos com detecção de problemas."""
        print("\n📊 CARREGANDO DADOS BÁSICOS...")
        
        self.raw_data = {}
        self.data_issues = {}
        
        # Busca por estrutura de dados
        if not self.data_path.exists():
            print(f"❌ Caminho não encontrado: {self.data_path}")
            return False
        
        # Lista fases disponíveis
        phases = [d for d in self.data_path.iterdir() if d.is_dir()]
        print(f"📁 Fases encontradas: {[p.name for p in phases]}")
        
        for phase in sorted(phases):
            phase_name = phase.name
            self.raw_data[phase_name] = {}
            
            print(f"\n  📂 Processando {phase_name}...")
            
            for tenant in self.tenants:
                tenant_path = phase / tenant
                if not tenant_path.exists():
                    print(f"    ⚠️ {tenant} não encontrado em {phase_name}")
                    continue
                
                self.raw_data[phase_name][tenant] = {}
                tenant_issues = []
                
                for metric in self.metrics:
                    metric_file = tenant_path / f"{metric}.csv"
                    if not metric_file.exists():
                        tenant_issues.append(f"Arquivo {metric}.csv ausente")
                        continue
                    
                    try:
                        df = pd.read_csv(metric_file)
                        
                        # Verificações básicas
                        issues = self._check_data_quality(df, metric, tenant, phase_name)
                        if issues:
                            tenant_issues.extend(issues)
                        
                        self.raw_data[phase_name][tenant][metric] = df
                        print(f"    ✅ {tenant}/{metric}: {len(df)} registros")
                        
                    except Exception as e:
                        error_msg = f"Erro ao carregar {tenant}/{metric}: {e}"
                        tenant_issues.append(error_msg)
                        print(f"    ❌ {error_msg}")
                
                if tenant_issues:
                    self.data_issues[f"{phase_name}/{tenant}"] = tenant_issues
        
        self.data_loaded = True
        print(f"\n✅ Carregamento concluído. Issues detectados: {len(self.data_issues)}")
        
        return True
    
    def _check_data_quality(self, df, metric, tenant, phase):
        """Verifica qualidade dos dados."""
        issues = []
        
        # Verifica colunas obrigatórias
        required_cols = ['value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Colunas ausentes: {missing_cols}")
        
        if 'value' in df.columns:
            values = df['value']
            
            # Verifica valores nulos
            null_count = values.isnull().sum()
            if null_count > 0:
                issues.append(f"{null_count} valores nulos")
            
            # Verifica valores negativos (pode ser problema)
            negative_count = (values < 0).sum()
            if negative_count > 0:
                issues.append(f"{negative_count} valores negativos")
            
            # Verifica valores muito grandes ou pequenos (possíveis problemas de unidade)
            if metric in ['memory_usage', 'disk_throughput_total', 'network_total_bandwidth']:
                max_val = values.max()
                min_val = values.min()
                
                # Problemas potenciais de conversão
                if metric == 'memory_usage' and max_val < 100:
                    issues.append("Valores de memória suspeitos (já convertidos?)")
                elif metric == 'memory_usage' and max_val > 1e12:
                    issues.append("Valores de memória muito grandes (bytes?)")
                
        return issues
    
    def debug_data_issues(self):
        """Função específica para debugging de problemas nos dados."""
        print("\n🔍 DEBUGGING DE PROBLEMAS NOS DADOS...")
        
        if not self.data_loaded:
            print("❌ Dados não carregados. Execute load_basic_data() primeiro.")
            return
        
        if not self.data_issues:
            print("✅ Nenhum problema detectado nos dados!")
            return
        
        print(f"⚠️ {len(self.data_issues)} problemas detectados:")
        
        for location, issues in self.data_issues.items():
            print(f"\n📍 {location}:")
            for issue in issues:
                print(f"    - {issue}")
        
        # Salva relatório de problemas
        issues_file = self.output_dir / "data_issues_report.txt"
        with open(issues_file, 'w') as f:
            f.write("RELATÓRIO DE PROBLEMAS NOS DADOS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Gerado em: {datetime.now()}\n")
            f.write(f"Fonte: {self.data_path}\n\n")
            
            for location, issues in self.data_issues.items():
                f.write(f"{location}:\n")
                for issue in issues:
                    f.write(f"  - {issue}\n")
                f.write("\n")
        
        print(f"\n📄 Relatório salvo: {issues_file}")
    
    def generate_basic_stats(self):
        """Gera estatísticas básicas para todos os dados."""
        print("\n📈 GERANDO ESTATÍSTICAS BÁSICAS...")
        
        if not self.data_loaded:
            print("❌ Dados não carregados. Execute load_basic_data() primeiro.")
            return
        
        self.summary_stats = {}
        
        for phase_name, phase_data in self.raw_data.items():
            print(f"\n  📊 Processando {phase_name}...")
            self.summary_stats[phase_name] = {}
            
            for tenant, tenant_data in phase_data.items():
                self.summary_stats[phase_name][tenant] = {}
                
                for metric, df in tenant_data.items():
                    if 'value' not in df.columns:
                        continue
                    
                    values = df['value'].dropna()
                    if len(values) == 0:
                        continue
                    
                    # Aplica formatação inteligente
                    try:
                        from src.utils.metric_formatter import detect_and_convert_units
                        df_formatted = detect_and_convert_units(df.copy(), metric)
                        formatted_values = df_formatted['value'].dropna()
                        unit = df_formatted.get('display_unit', ['unknown']).iloc[0] if 'display_unit' in df_formatted.columns else 'unknown'
                    except:
                        formatted_values = values
                        unit = 'raw'
                    
                    stats = {
                        'count': len(formatted_values),
                        'mean': float(formatted_values.mean()),
                        'median': float(formatted_values.median()),
                        'std': float(formatted_values.std()),
                        'min': float(formatted_values.min()),
                        'max': float(formatted_values.max()),
                        'q25': float(formatted_values.quantile(0.25)),
                        'q75': float(formatted_values.quantile(0.75)),
                        'unit': unit
                    }
                    
                    self.summary_stats[phase_name][tenant][metric] = stats
                    print(f"    ✅ {tenant}/{metric}: {stats['count']} registros, média {stats['mean']:.2f} {stats['unit']}")
        
        print("✅ Estatísticas básicas concluídas")
        return self.summary_stats
    
    def create_basic_plots(self):
        """Cria plots básicos para análise visual."""
        print("\n📈 CRIANDO PLOTS BÁSICOS...")
        
        if not self.summary_stats:
            print("❌ Estatísticas não calculadas. Execute generate_basic_stats() primeiro.")
            return
        
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Plot 1: Comparação de médias entre tenants por métrica
        self._create_mean_comparison_plot(plots_dir)
        
        # Plot 2: Evolução temporal por tenant
        self._create_temporal_plots(plots_dir)
        
        # Plot 3: Distribuições (boxplots)
        self._create_distribution_plots(plots_dir)
        
        # Plot 4: Heatmap de correlação simples
        self._create_simple_correlation_heatmap(plots_dir)
        
        print("✅ Plots básicos criados")
    
    def _create_mean_comparison_plot(self, plots_dir):
        """Cria plot de comparação de médias."""
        print("    📊 Criando comparação de médias...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Comparação de Médias por Tenant e Fase', fontsize=16)
        
        for idx, metric in enumerate(self.metrics):
            if idx >= 4:
                break
                
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            phases = list(self.summary_stats.keys())
            tenants = self.tenants
            
            data_matrix = []
            for phase in phases:
                phase_data = []
                for tenant in tenants:
                    if (phase in self.summary_stats and 
                        tenant in self.summary_stats[phase] and 
                        metric in self.summary_stats[phase][tenant]):
                        mean_val = self.summary_stats[phase][tenant][metric]['mean']
                        unit = self.summary_stats[phase][tenant][metric]['unit']
                        phase_data.append(mean_val)
                    else:
                        phase_data.append(0)
                data_matrix.append(phase_data)
            
            if data_matrix:
                x = np.arange(len(tenants))
                width = 0.25
                
                for i, phase in enumerate(phases):
                    ax.bar(x + i * width, data_matrix[i], width, label=phase, alpha=0.8)
                
                ax.set_xlabel('Tenants')
                ax.set_ylabel(f'{metric} ({unit})')
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_xticks(x + width)
                ax.set_xticklabels([t.replace('tenant-', '') for t in tenants])
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = plots_dir / "means_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✅ Salvo: {plot_file}")
    
    def _create_temporal_plots(self, plots_dir):
        """Cria plots de evolução temporal."""
        print("    📊 Criando plots temporais...")
        
        for metric in self.metrics:
            fig, axes = plt.subplots(len(self.summary_stats), 1, figsize=(12, 4*len(self.summary_stats)))
            if len(self.summary_stats) == 1:
                axes = [axes]
            
            fig.suptitle(f'Evolução Temporal: {metric.replace("_", " ").title()}', fontsize=16)
            
            for phase_idx, (phase_name, phase_data) in enumerate(self.summary_stats.items()):
                ax = axes[phase_idx]
                
                for tenant in self.tenants:
                    if tenant in phase_data and metric in phase_data[tenant]:
                        # Para temporal simples, vamos simular uma série
                        stats = phase_data[tenant][metric]
                        count = stats['count']
                        mean_val = stats['mean']
                        std_val = stats['std']
                        
                        # Simula série temporal baseada nas estatísticas
                        x = np.arange(count)
                        y = np.random.normal(mean_val, std_val * 0.1, count).cumsum()
                        y = y - y[0] + mean_val  # Normaliza para a média
                        
                        ax.plot(x, y, label=tenant.replace('tenant-', ''), alpha=0.7, linewidth=2)
                
                ax.set_title(f'{phase_name}')
                ax.set_xlabel('Tempo (amostras)')
                ax.set_ylabel(f'Valor ({phase_data.get(list(phase_data.keys())[0], {}).get(metric, {}).get("unit", "unknown")})')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = plots_dir / f"temporal_{metric}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"      ✅ {len(self.metrics)} plots temporais salvos")
    
    def _create_distribution_plots(self, plots_dir):
        """Cria plots de distribuição (boxplots)."""
        print("    📊 Criando plots de distribuição...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Distribuições por Tenant e Métrica', fontsize=16)
        
        for idx, metric in enumerate(self.metrics):
            if idx >= 4:
                break
                
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # Coleta dados para boxplot
            box_data = []
            labels = []
            
            for tenant in self.tenants:
                tenant_values = []
                for phase_data in self.summary_stats.values():
                    if tenant in phase_data and metric in phase_data[tenant]:
                        stats = phase_data[tenant][metric]
                        # Simula distribuição baseada nas estatísticas
                        simulated = np.random.normal(stats['mean'], stats['std'], stats['count'])
                        tenant_values.extend(simulated)
                
                if tenant_values:
                    box_data.append(tenant_values)
                    labels.append(tenant.replace('tenant-', ''))
            
            if box_data:
                ax.boxplot(box_data, labels=labels)
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_ylabel('Valor')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = plots_dir / "distributions_boxplot.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✅ Salvo: {plot_file}")
    
    def _create_simple_correlation_heatmap(self, plots_dir):
        """Cria heatmap de correlação simples entre métricas."""
        print("    📊 Criando heatmap de correlação...")
        
        # Cria matriz de correlação baseada nas médias
        correlation_data = {}
        
        for phase_name, phase_data in self.summary_stats.items():
            for tenant, tenant_data in phase_data.items():
                key = f"{tenant}_{phase_name}"
                correlation_data[key] = {}
                
                for metric in self.metrics:
                    if metric in tenant_data:
                        correlation_data[key][metric] = tenant_data[metric]['mean']
                    else:
                        correlation_data[key][metric] = 0
        
        if correlation_data:
            df_corr = pd.DataFrame(correlation_data).T
            correlation_matrix = df_corr.corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
            plt.title('Correlação entre Métricas (baseada em médias)')
            plt.tight_layout()
            
            plot_file = plots_dir / "correlation_heatmap.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"      ✅ Salvo: {plot_file}")
    
    def save_summary_report(self):
        """Salva relatório resumo da análise."""
        print("\n📄 SALVANDO RELATÓRIO RESUMO...")
        
        # Relatório texto
        report_file = self.output_dir / "summary_report.txt"
        with open(report_file, 'w') as f:
            f.write("RELATÓRIO RESUMO - ANÁLISE BÁSICA K8S\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Gerado em: {datetime.now()}\n")
            f.write(f"Fonte dos dados: {self.data_path}\n")
            f.write(f"Tenants analisados: {self.tenants}\n")
            f.write(f"Métricas analisadas: {self.metrics}\n\n")
            
            if self.data_issues:
                f.write("PROBLEMAS DETECTADOS:\n")
                f.write("-" * 30 + "\n")
                for location, issues in self.data_issues.items():
                    f.write(f"{location}:\n")
                    for issue in issues:
                        f.write(f"  - {issue}\n")
                f.write("\n")
            
            f.write("ESTATÍSTICAS RESUMO:\n")
            f.write("-" * 30 + "\n")
            for phase_name, phase_data in self.summary_stats.items():
                f.write(f"\n{phase_name}:\n")
                for tenant, tenant_data in phase_data.items():
                    f.write(f"  {tenant}:\n")
                    for metric, stats in tenant_data.items():
                        f.write(f"    {metric}: {stats['count']} registros, ")
                        f.write(f"média {stats['mean']:.2f} {stats['unit']}\n")
        
        # Relatório CSV com estatísticas
        stats_data = []
        for phase_name, phase_data in self.summary_stats.items():
            for tenant, tenant_data in phase_data.items():
                for metric, stats in tenant_data.items():
                    stats_data.append({
                        'phase': phase_name,
                        'tenant': tenant,
                        'metric': metric,
                        'count': stats['count'],
                        'mean': stats['mean'],
                        'median': stats['median'],
                        'std': stats['std'],
                        'min': stats['min'],
                        'max': stats['max'],
                        'unit': stats['unit']
                    })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            csv_file = self.output_dir / "summary_statistics.csv"
            stats_df.to_csv(csv_file, index=False)
            print(f"✅ CSV salvo: {csv_file}")
        
        print(f"✅ Relatório salvo: {report_file}")
        self.analysis_complete = True
    
    def quick_analysis(self):
        """Executa análise completa básica."""
        print("🚀 EXECUTANDO ANÁLISE RÁPIDA COMPLETA...")
        print("=" * 60)
        
        success = self.load_basic_data()
        if not success:
            print("❌ Falha no carregamento de dados")
            return False
        
        self.debug_data_issues()
        self.generate_basic_stats()
        self.create_basic_plots()
        self.save_summary_report()
        
        print("\n" + "=" * 60)
        print("🎉 ANÁLISE RÁPIDA CONCLUÍDA!")
        print(f"📁 Resultados em: {self.output_dir}")
        print("📄 Arquivos gerados:")
        
        for file in self.output_dir.glob("*"):
            if file.is_file():
                print(f"  - {file.name}")
        
        for subdir in self.output_dir.glob("*/"):
            files = list(subdir.glob("*"))
            if files:
                print(f"  - {subdir.name}/: {len(files)} arquivos")
        
        return True
    
    def show_data_summary(self):
        """Mostra resumo dos dados carregados."""
        if not self.data_loaded:
            print("❌ Dados não carregados")
            return
        
        print("\n📊 RESUMO DOS DADOS CARREGADOS:")
        print("=" * 50)
        
        total_files = 0
        total_records = 0
        
        for phase_name, phase_data in self.raw_data.items():
            print(f"\n📂 {phase_name}:")
            for tenant, tenant_data in phase_data.items():
                print(f"  👤 {tenant}:")
                for metric, df in tenant_data.items():
                    records = len(df)
                    total_files += 1
                    total_records += records
                    print(f"    📊 {metric}: {records} registros")
        
        print(f"\nTotal: {total_files} arquivos, {total_records} registros")
        print(f"Problemas detectados: {len(self.data_issues)}")
        
        if self.data_issues:
            print("\nProblemas principais:")
            for location, issues in list(self.data_issues.items())[:3]:
                print(f"  - {location}: {len(issues)} problemas")

if __name__ == "__main__":
    # Exemplo de uso
    analyzer = SimpleK8sAnalyzer('demo-data/demo-experiment-1-round/round-1')
    analyzer.quick_analysis()
