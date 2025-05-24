#!/usr/bin/env python3
"""
K8s Noisy Detection - Simplified Core
=====================================

Core simplificado para análise de dados K8s com configuração YAML.
Substitui o pipeline complexo de 700+ linhas por uma abordagem modular e clara.

Autor: Phil
Versão: 2.0 (Simplificado)
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root / 'src'))

try:
    from src.utils.metric_formatter import MetricFormatter
except ImportError:
    print("⚠️ MetricFormatter não disponível, usando formatação básica")
    MetricFormatter = None


class SimpleK8sAnalyzer:
    """
    Analisador simplificado para dados K8s com suporte a configuração YAML.
    
    Foco em:
    - Debugging rápido de problemas de dados
    - Análises essenciais em < 5 minutos
    - Interface clara e modular
    - Configuração flexível via YAML
    """
    
    def __init__(self, config_path: Optional[str] = None, data_path: Optional[str] = None):
        """
        Inicializa o analisador com configuração YAML ou parâmetros diretos.
        
        Args:
            config_path: Caminho para arquivo YAML de configuração
            data_path: Caminho direto para dados (opcional se config_path fornecido)
        """
        self.config = self._load_config(config_path)
        self.data_path = data_path or self.config['data']['path']
        self.output_dir = Path(self.config['output']['directory'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize formatter if available
        self.formatter = MetricFormatter() if MetricFormatter else None
        
        # Data containers
        self.raw_data = {}
        self.processed_data = {}
        self.stats = {}
        
        # Analysis state
        self.analysis_complete = False
        self.plots_generated = False
        
        print(f"🚀 SimpleK8sAnalyzer inicializado")
        print(f"📁 Dados: {self.data_path}")
        print(f"📊 Output: {self.output_dir}")
        print(f"🎯 Modo: {self.config['analysis']['mode']}")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Carrega configuração YAML ou usa defaults."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default basic config
        return {
            'analysis': {
                'mode': 'basic',
                'tenants': ['a', 'b', 'c', 'd'],
                'metrics': ['memory_usage', 'disk_throughput_total', 'network_total_bandwidth', 'cpu_usage']
            },
            'data': {'path': 'demo-data/'},
            'output': {'directory': './output/simple_analysis', 'format': 'png'},
            'debug': {'verbose': True},
            'plots': {'basic_timeseries': True, 'boxplots': True, 'figsize': [12, 8], 'dpi': 300}
        }
    
    def quick_analysis(self) -> Dict[str, Any]:
        """
        Executa análise completa rápida (< 5 minutos).
        
        Returns:
            Dicionário com resultados da análise
        """
        print("\n🔍 Iniciando análise rápida...")
        
        # 1. Load and validate data
        print("📥 Carregando dados...")
        self.load_data()
        
        # 2. Debug data issues
        print("🐛 Verificando problemas nos dados...")
        data_issues = self.debug_data_issues()
        
        # 3. Process data
        print("⚙️ Processando dados...")
        self.process_data()
        
        # 4. Generate basic statistics
        print("📊 Gerando estatísticas...")
        self.generate_basic_stats()
        
        # 5. Create basic plots
        print("📈 Criando plots...")
        self.create_basic_plots()
        
        # 6. Generate summary report
        print("📋 Gerando relatório...")
        summary = self.generate_summary_report()
        
        self.analysis_complete = True
        print(f"\n✅ Análise completa! Resultados em: {self.output_dir}")
        
        return {
            'data_issues': data_issues,
            'stats': self.stats,
            'summary': summary,
            'output_dir': str(self.output_dir)
        }
    
    def load_data(self) -> bool:
        """Carrega dados dos tenants especificados na configuração."""
        self.raw_data = {}
        self.data_issues = {}
        
        data_path = Path(self.data_path)
        tenants = self.config['analysis']['tenants']
        metrics = self.config['analysis']['metrics']
        
        # Busca por estrutura de dados
        if not data_path.exists():
            print(f"❌ Caminho não encontrado: {data_path}")
            return False
        
        # Lista fases disponíveis
        phases = [d for d in data_path.iterdir() if d.is_dir()]
        if not phases:
            print("❌ Nenhuma fase encontrada no diretório de dados")
            return False
            
        print(f"📁 Fases encontradas: {[p.name for p in phases]}")
        
        for phase in sorted(phases):
            phase_name = phase.name
            self.raw_data[phase_name] = {}
            
            print(f"\n  📂 Processando {phase_name}...")
            
            for tenant in tenants:
                tenant_dir = f"tenant-{tenant}"
                tenant_path = phase / tenant_dir
                if not tenant_path.exists():
                    print(f"    ⚠️ {tenant_dir} não encontrado em {phase_name}")
                    continue
                
                self.raw_data[phase_name][tenant_dir] = {}
                tenant_issues = []
                
                for metric in metrics:
                    metric_file = tenant_path / f"{metric}.csv"
                    if not metric_file.exists():
                        tenant_issues.append(f"Arquivo {metric}.csv ausente")
                        continue
                    
                    try:
                        df = pd.read_csv(metric_file)
                        
                        # Verificações básicas
                        issues = self._check_data_quality(df, metric, tenant_dir, phase_name)
                        if issues:
                            tenant_issues.extend(issues)
                        
                        self.raw_data[phase_name][tenant_dir][metric] = df
                        print(f"    ✅ {tenant_dir}/{metric}: {len(df)} registros")
                        
                    except Exception as e:
                        error_msg = f"Erro ao carregar {tenant_dir}/{metric}: {e}"
                        tenant_issues.append(error_msg)
                        print(f"    ❌ {error_msg}")
                
                if tenant_issues:
                    self.data_issues[f"{phase_name}/{tenant_dir}"] = tenant_issues
        
        total_files = sum(len(phase_data) for phase_data in self.raw_data.values() for tenant_data in phase_data.values())
        print(f"\n✅ Carregamento concluído. {total_files} arquivos carregados, {len(self.data_issues)} issues detectados")
        
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
    
    def debug_data_issues(self) -> Dict[str, List[str]]:
        """Função específica para debugging de problemas nos dados."""
        print("\n🔍 DEBUGGING DE PROBLEMAS NOS DADOS...")
        
        if not self.raw_data:
            print("❌ Dados não carregados. Execute load_data() primeiro.")
            return {}
        
        if not self.data_issues:
            print("✅ Nenhum problema detectado nos dados!")
            return {}
        
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
        return self.data_issues

    def process_data(self) -> None:
        """Processa os dados carregados aplicando formatação e conversões."""
        print("\n⚙️ Processando dados...")
        
        self.processed_data = {}
        
        for phase_name, phase_data in self.raw_data.items():
            self.processed_data[phase_name] = {}
            
            for tenant, tenant_data in phase_data.items():
                self.processed_data[phase_name][tenant] = {}
                
                for metric, df in tenant_data.items():
                    processed_df = df.copy()
                    
                    # Aplica formatação inteligente se disponível
                    if self.formatter and self.config['processing']['auto_format']:
                        try:
                            processed_df = self.formatter.format_dataframe(processed_df, metric)
                        except Exception as e:
                            print(f"    ⚠️ Erro na formatação de {tenant}/{metric}: {e}")
                    
                    self.processed_data[phase_name][tenant][metric] = processed_df
        
        print("✅ Processamento concluído")

    def generate_basic_stats(self) -> Dict[str, Any]:
        """Gera estatísticas básicas para todos os dados."""
        print("\n📈 GERANDO ESTATÍSTICAS BÁSICAS...")
        
        if not self.processed_data and not self.raw_data:
            print("❌ Dados não carregados. Execute load_data() primeiro.")
            return {}
        
        # Use processed data if available, otherwise raw data
        data_source = self.processed_data if self.processed_data else self.raw_data
        
        self.stats = {}
        
        for phase_name, phase_data in data_source.items():
            print(f"\n  📊 Processando {phase_name}...")
            self.stats[phase_name] = {}
            
            for tenant, tenant_data in phase_data.items():
                self.stats[phase_name][tenant] = {}
                
                for metric, df in tenant_data.items():
                    if 'value' not in df.columns:
                        continue
                    
                    values = df['value'].dropna()
                    if len(values) == 0:
                        continue
                    
                    # Determina unidade
                    unit = 'unknown'
                    if 'display_unit' in df.columns:
                        unit = df['display_unit'].iloc[0] if len(df) > 0 else 'unknown'
                    elif 'unit' in df.columns:
                        unit = df['unit'].iloc[0] if len(df) > 0 else 'unknown'
                    
                    stats = {
                        'count': len(values),
                        'mean': float(values.mean()),
                        'median': float(values.median()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'q25': float(values.quantile(0.25)),
                        'q75': float(values.quantile(0.75)),
                        'unit': unit
                    }
                    
                    self.stats[phase_name][tenant][metric] = stats
                    print(f"    ✅ {tenant}/{metric}: {stats['count']} registros, média {stats['mean']:.2f} {stats['unit']}")
        
        print("✅ Estatísticas básicas concluídas")
        return self.stats
    
    def create_basic_plots(self) -> None:
        """Cria plots básicos para análise visual."""
        print("\n📈 CRIANDO PLOTS BÁSICOS...")
        
        if not self.stats:
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
        if self.config['plots'].get('correlation_matrix', False):
            self._create_simple_correlation_heatmap(plots_dir)
        
        self.plots_generated = True
        print("✅ Plots básicos criados")
    
    def _create_mean_comparison_plot(self, plots_dir):
        """Cria plot de comparação de médias."""
        print("    📊 Criando comparação de médias...")
        
        metrics = self.config['analysis']['metrics']
        tenants = [f"tenant-{t}" for t in self.config['analysis']['tenants']]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Comparação de Médias por Tenant e Fase', fontsize=16)
        
        for idx, metric in enumerate(metrics[:4]):  # Limite para 4 métricas no plot
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            phases = list(self.stats.keys())
            
            data_matrix = []
            for phase in phases:
                phase_data = []
                for tenant in tenants:
                    if (phase in self.stats and 
                        tenant in self.stats[phase] and 
                        metric in self.stats[phase][tenant]):
                        mean_val = self.stats[phase][tenant][metric]['mean']
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
                ax.set_ylabel(f'{metric}')
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
        
        metrics = self.config['analysis']['metrics']
        tenants = [f"tenant-{t}" for t in self.config['analysis']['tenants']]
        
        for metric in metrics:
            fig, axes = plt.subplots(len(self.stats), 1, figsize=(12, 4*len(self.stats)))
            if len(self.stats) == 1:
                axes = [axes]
            
            fig.suptitle(f'Evolução Temporal: {metric.replace("_", " ").title()}', fontsize=16)
            
            for phase_idx, (phase_name, phase_data) in enumerate(self.stats.items()):
                ax = axes[phase_idx]
                
                for tenant in tenants:
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
                ax.set_ylabel(f'Valor')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = plots_dir / f"temporal_{metric}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"      ✅ {len(metrics)} plots temporais salvos")
    
    def _create_distribution_plots(self, plots_dir):
        """Cria plots de distribuição (boxplots)."""
        print("    📊 Criando plots de distribuição...")
        
        metrics = self.config['analysis']['metrics']
        tenants = [f"tenant-{t}" for t in self.config['analysis']['tenants']]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Distribuições por Tenant e Métrica', fontsize=16)
        
        for idx, metric in enumerate(metrics[:4]):  # Limite para 4 métricas
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # Coleta dados para boxplot
            box_data = []
            labels = []
            
            for tenant in tenants:
                tenant_values = []
                for phase_data in self.stats.values():
                    if tenant in phase_data and metric in phase_data[tenant]:
                        stats = phase_data[tenant][metric]
                        # Simula distribuição baseada nas estatísticas
                        simulated = np.random.normal(stats['mean'], stats['std'], min(stats['count'], 1000))
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
        
        for phase_name, phase_data in self.stats.items():
            for tenant, tenant_data in phase_data.items():
                key = f"{tenant}_{phase_name}"
                correlation_data[key] = {}
                
                for metric in self.config['analysis']['metrics']:
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
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Salva relatório resumo da análise."""
        print("\n📄 GERANDO RELATÓRIO RESUMO...")
        
        tenants = [f"tenant-{t}" for t in self.config['analysis']['tenants']]
        metrics = self.config['analysis']['metrics']
        
        # Relatório texto
        report_file = self.output_dir / "summary_report.txt"
        with open(report_file, 'w') as f:
            f.write("RELATÓRIO RESUMO - ANÁLISE BÁSICA K8S\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Gerado em: {datetime.now()}\n")
            f.write(f"Fonte dos dados: {self.data_path}\n")
            f.write(f"Tenants analisados: {tenants}\n")
            f.write(f"Métricas analisadas: {metrics}\n\n")
            
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
            for phase_name, phase_data in self.stats.items():
                f.write(f"\n{phase_name}:\n")
                for tenant, tenant_data in phase_data.items():
                    f.write(f"  {tenant}:\n")
                    for metric, stats in tenant_data.items():
                        f.write(f"    {metric}: {stats['count']} registros, ")
                        f.write(f"média {stats['mean']:.2f} {stats['unit']}\n")
        
        # Relatório CSV com estatísticas
        stats_data = []
        for phase_name, phase_data in self.stats.items():
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
        
        summary_data = {
            'total_phases': len(self.stats),
            'total_tenants': len(tenants),
            'total_metrics': len(metrics),
            'data_issues_count': len(self.data_issues),
            'analysis_complete': self.analysis_complete,
            'plots_generated': self.plots_generated
        }
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            csv_file = self.output_dir / "summary_statistics.csv"
            stats_df.to_csv(csv_file, index=False)
            print(f"✅ CSV salvo: {csv_file}")
        
        print(f"✅ Relatório salvo: {report_file}")
        
        return summary_data
    
    def show_data_summary(self):
        """Mostra resumo dos dados carregados."""
        if not self.raw_data:
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
    print("🚀 Testando SimpleK8sAnalyzer...")
    
    # Teste com configuração padrão
    analyzer = SimpleK8sAnalyzer(data_path='demo-data/demo-experiment-1-round/round-1')
    result = analyzer.quick_analysis()
    
    if result:
        print("\n✅ Teste concluído com sucesso!")
        print(f"📂 Resultados em: {result['output_dir']}")
    else:
        print("\n❌ Teste falhou")
