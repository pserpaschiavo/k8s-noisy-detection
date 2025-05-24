#!/usr/bin/env python3
"""
CLI para Geração de Plots e Tabelas - K8s Noisy Detection
Sistema de Formatação Inteligente de Métricas
"""

import sys
import os
sys.path.insert(0, '.')

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

def setup_cli():
    """Configura argumentos da CLI."""
    parser = argparse.ArgumentParser(
        description='Gerador de plots e tabelas para análise K8s',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python cli_plots.py --tenants a,b,c,d --metrics memory,network --phases all
  python cli_plots.py --list-data
  python cli_plots.py --quick-test
  python cli_plots.py --comparison --output ./results
        """
    )
    
    # Argumentos principais
    parser.add_argument('--tenants', '-t', 
                       help='Tenants a analisar (ex: a,b,c,d ou tenant-a,tenant-b)')
    parser.add_argument('--metrics', '-m',
                       help='Métricas a processar (ex: memory,disk,network,cpu)')
    parser.add_argument('--phases', '-p',
                       help='Fases a analisar (ex: baseline,attack,recovery ou all)')
    parser.add_argument('--output', '-o', default='./cli_output',
                       help='Diretório de saída (padrão: ./cli_output)')
    
    # Opções de dados
    parser.add_argument('--data-path', default='demo-data/demo-experiment-1-round/round-1',
                       help='Caminho para os dados')
    parser.add_argument('--experiment', choices=['1-round', '3-rounds'], default='1-round',
                       help='Experimento a usar')
    
    # Modos de operação
    parser.add_argument('--list-data', action='store_true',
                       help='Lista dados disponíveis')
    parser.add_argument('--quick-test', action='store_true',
                       help='Teste rápido com dados padrão')
    parser.add_argument('--comparison', action='store_true',
                       help='Gera comparação entre tenants')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Modo interativo')
    
    # Opções de formatação
    parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png',
                       help='Formato dos plots')
    parser.add_argument('--style', choices=['default', 'seaborn', 'ggplot'], default='seaborn',
                       help='Estilo dos plots')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Output verboso')
    
    return parser

def list_available_data(data_path):
    """Lista dados disponíveis."""
    print("🔍 DADOS DISPONÍVEIS")
    print("=" * 50)
    
    base_path = Path(data_path)
    if not base_path.exists():
        print(f"❌ Caminho não encontrado: {data_path}")
        return
    
    # Lista fases
    phases = [d for d in base_path.iterdir() if d.is_dir()]
    print(f"📁 Caminho base: {base_path}")
    print(f"📊 Fases encontradas: {len(phases)}")
    
    for phase in sorted(phases):
        print(f"\n  📁 {phase.name}")
        
        # Lista tenants
        tenants = [d for d in phase.iterdir() if d.is_dir()]
        for tenant in sorted(tenants):
            print(f"    👤 {tenant.name}")
            
            # Lista métricas
            metrics = [f for f in tenant.iterdir() if f.suffix == '.csv']
            for metric in sorted(metrics):
                try:
                    df = pd.read_csv(metric)
                    print(f"      📊 {metric.name}: {len(df)} registros")
                except:
                    print(f"      ❌ {metric.name}: erro ao ler")

def load_tenant_data(base_path, tenant, phases=None):
    """Carrega dados de um tenant."""
    tenant_data = {}
    base_path = Path(base_path)
    
    # Define fases a carregar
    if phases is None or phases == 'all':
        phase_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    else:
        phase_mapping = {
            'baseline': '1 - Baseline',
            'attack': '2 - Attack', 
            'recovery': '3 - Recovery'
        }
        phase_dirs = [base_path / phase_mapping.get(p, p) for p in phases]
    
    for phase_dir in sorted(phase_dirs):
        if not phase_dir.exists():
            continue
            
        phase_name = phase_dir.name
        tenant_path = phase_dir / tenant
        
        if not tenant_path.exists():
            continue
            
        phase_data = {}
        metrics = [f for f in tenant_path.iterdir() if f.suffix == '.csv']
        
        for metric_file in metrics:
            try:
                df = pd.read_csv(metric_file)
                metric_name = metric_file.stem
                
                # Aplica formatação inteligente
                try:
                    from src.utils.metric_formatter import detect_and_convert_units
                    df_formatted = detect_and_convert_units(df.copy(), metric_name)
                    phase_data[metric_name] = df_formatted
                except:
                    phase_data[metric_name] = df
                    
            except Exception as e:
                print(f"⚠️ Erro ao carregar {tenant}/{phase_name}/{metric_file.name}: {e}")
        
        if phase_data:
            tenant_data[phase_name] = phase_data
    
    return tenant_data

def create_metric_plot(data_dict, metric_name, output_path, plot_format='png', style='seaborn'):
    """Cria plot para uma métrica específica."""
    plt.style.use(style)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Comparação: {metric_name.replace("_", " ").title()}', fontsize=16)
    
    tenant_names = list(data_dict.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(tenant_names)))
    
    for idx, (tenant, tenant_data) in enumerate(data_dict.items()):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col] if len(tenant_names) > 1 else axes
        
        # Combina dados de todas as fases
        all_values = []
        all_timestamps = []
        phase_labels = []
        
        for phase_name, phase_data in tenant_data.items():
            if metric_name in phase_data:
                df = phase_data[metric_name]
                if 'value' in df.columns:
                    values = df['value'].dropna()
                    timestamps = range(len(all_values), len(all_values) + len(values))
                    
                    all_values.extend(values)
                    all_timestamps.extend(timestamps)
                    phase_labels.extend([phase_name] * len(values))
        
        if all_values:
            # Plot principal
            ax.plot(all_timestamps, all_values, color=colors[idx], alpha=0.7, linewidth=1.5)
            ax.scatter(all_timestamps[::10], all_values[::10], color=colors[idx], s=20, alpha=0.8)
            
            # Estatísticas
            stats_text = f'Média: {np.mean(all_values):.2f}\n'
            stats_text += f'Max: {np.max(all_values):.2f}\n'
            stats_text += f'Min: {np.min(all_values):.2f}'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f'{tenant}')
        ax.set_xlabel('Tempo (amostras)')
        ax.set_ylabel('Valor')
        ax.grid(True, alpha=0.3)
    
    # Remove subplots vazios
    for idx in range(len(tenant_names), 4):
        row = idx // 2
        col = idx % 2
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    
    output_file = output_path / f'{metric_name}_comparison.{plot_format}'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def create_summary_table(data_dict, metric_name, output_path):
    """Cria tabela resumo para uma métrica."""
    summary_data = []
    
    for tenant, tenant_data in data_dict.items():
        for phase_name, phase_data in tenant_data.items():
            if metric_name in phase_data:
                df = phase_data[metric_name]
                if 'value' in df.columns:
                    values = df['value'].dropna()
                    
                    if len(values) > 0:
                        # Detecta unidade se disponível
                        unit = df.get('display_unit', ['unknown']).iloc[0] if 'display_unit' in df.columns else 'unknown'
                        
                        summary_data.append({
                            'Tenant': tenant,
                            'Fase': phase_name,
                            'Registros': len(values),
                            'Média': f"{np.mean(values):.2f}",
                            'Mediana': f"{np.median(values):.2f}",
                            'Desvio Padrão': f"{np.std(values):.2f}",
                            'Mínimo': f"{np.min(values):.2f}",
                            'Máximo': f"{np.max(values):.2f}",
                            'Unidade': unit
                        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Salva CSV
        csv_file = output_path / f'{metric_name}_summary.csv'
        summary_df.to_csv(csv_file, index=False)
        
        # Salva tabela formatada
        txt_file = output_path / f'{metric_name}_summary.txt'
        with open(txt_file, 'w') as f:
            f.write(f"RESUMO: {metric_name.replace('_', ' ').title()}\n")
            f.write("=" * 60 + "\n\n")
            f.write(summary_df.to_string(index=False))
            f.write(f"\n\nGerado em: {datetime.now()}\n")
        
        return csv_file, txt_file
    
    return None, None

def interactive_mode():
    """Modo interativo para seleção de opções."""
    print("🎯 MODO INTERATIVO - Geração de Plots e Tabelas")
    print("=" * 60)
    
    # Seleciona experimento
    print("\n📊 Selecione o experimento:")
    print("1. Demo 1-round (demo-experiment-1-round)")
    print("2. Demo 3-rounds (demo-experiment-3-rounds)")
    
    exp_choice = input("Escolha (1-2): ").strip()
    
    if exp_choice == "2":
        data_base = "demo-data/demo-experiment-3-rounds"
        print("3 rounds disponíveis. Selecione o round:")
        print("1. round-1")
        print("2. round-2") 
        print("3. round-3")
        round_choice = input("Escolha (1-3): ").strip()
        data_path = f"{data_base}/round-{round_choice}"
    else:
        data_path = "demo-data/demo-experiment-1-round/round-1"
    
    # Seleciona tenants
    print(f"\n👥 Tenants disponíveis em {data_path}:")
    print("1. tenant-a")
    print("2. tenant-b")
    print("3. tenant-c")
    print("4. tenant-d")
    print("5. Todos (a,b,c,d)")
    
    tenant_choice = input("Escolha tenants (ex: 1,2,3 ou 5 para todos): ").strip()
    
    if "5" in tenant_choice:
        tenants = ["tenant-a", "tenant-b", "tenant-c", "tenant-d"]
    else:
        tenant_map = {"1": "tenant-a", "2": "tenant-b", "3": "tenant-c", "4": "tenant-d"}
        tenants = [tenant_map[c] for c in tenant_choice.split(",") if c in tenant_map]
    
    # Seleciona métricas
    print("\n📊 Métricas disponíveis:")
    print("1. memory_usage")
    print("2. disk_throughput_total")
    print("3. network_total_bandwidth")
    print("4. cpu_usage")
    print("5. Todas")
    
    metric_choice = input("Escolha métricas (ex: 1,2 ou 5 para todas): ").strip()
    
    if "5" in metric_choice:
        metrics = ["memory_usage", "disk_throughput_total", "network_total_bandwidth", "cpu_usage"]
    else:
        metric_map = {"1": "memory_usage", "2": "disk_throughput_total", 
                     "3": "network_total_bandwidth", "4": "cpu_usage"}
        metrics = [metric_map[c] for c in metric_choice.split(",") if c in metric_map]
    
    # Diretório de saída
    output_dir = input("\n📁 Diretório de saída (Enter para 'cli_output'): ").strip()
    if not output_dir:
        output_dir = "cli_output"
    
    return {
        'data_path': data_path,
        'tenants': tenants,
        'metrics': metrics,
        'output': output_dir,
        'phases': 'all'
    }

def main():
    parser = setup_cli()
    args = parser.parse_args()
    
    # Modo interativo
    if args.interactive:
        config = interactive_mode()
        args.data_path = config['data_path']
        args.tenants = ','.join(config['tenants'])
        args.metrics = ','.join(config['metrics'])
        args.output = config['output']
        args.phases = config['phases']
    
    # Lista dados disponíveis
    if args.list_data:
        list_available_data(args.data_path)
        return
    
    # Teste rápido
    if args.quick_test:
        print("🚀 TESTE RÁPIDO")
        print("=" * 30)
        print("Processando tenant-a com memory_usage...")
        
        try:
            data = load_tenant_data(args.data_path, 'tenant-a')
            if data:
                output_path = Path(args.output)
                output_path.mkdir(exist_ok=True)
                
                data_dict = {'tenant-a': data}
                
                plot_file = create_metric_plot(data_dict, 'memory_usage', output_path, args.format, args.style)
                csv_file, txt_file = create_summary_table(data_dict, 'memory_usage', output_path)
                
                print(f"✅ Plot salvo: {plot_file}")
                if csv_file:
                    print(f"✅ Tabela CSV: {csv_file}")
                    print(f"✅ Resumo TXT: {txt_file}")
                
                print("\n🎉 Teste rápido concluído!")
            else:
                print("❌ Nenhum dado encontrado para tenant-a")
        except Exception as e:
            print(f"❌ Erro no teste rápido: {e}")
        return
    
    # Processamento principal
    if not args.tenants or not args.metrics:
        print("❌ Especifique --tenants e --metrics ou use --interactive")
        print("Use --help para ver opções disponíveis")
        return
    
    # Parse argumentos
    tenants = [t.strip() for t in args.tenants.split(',')]
    metrics = [m.strip() for m in args.metrics.split(',')]
    phases = args.phases.split(',') if args.phases != 'all' else 'all'
    
    # Cria diretório de saída
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)
    
    print("🎯 GERAÇÃO DE PLOTS E TABELAS VIA CLI")
    print("=" * 50)
    print(f"📁 Dados: {args.data_path}")
    print(f"👥 Tenants: {tenants}")
    print(f"📊 Métricas: {metrics}")
    print(f"📁 Saída: {output_path}")
    print(f"🎨 Formato: {args.format}")
    print(f"🎨 Estilo: {args.style}")
    
    # Carrega dados
    print(f"\n📊 CARREGANDO DADOS...")
    all_data = {}
    
    for tenant in tenants:
        if args.verbose:
            print(f"  👤 Carregando {tenant}...")
        
        tenant_data = load_tenant_data(args.data_path, tenant, phases)
        if tenant_data:
            all_data[tenant] = tenant_data
            if args.verbose:
                total_records = sum(len(phase_data.get(m, [])) for phase_data in tenant_data.values() for m in metrics)
                print(f"    ✅ {len(tenant_data)} fases, ~{total_records} registros")
        else:
            print(f"    ⚠️ Nenhum dado encontrado para {tenant}")
    
    if not all_data:
        print("❌ Nenhum dado carregado. Verifique os caminhos e nomes dos tenants.")
        return
    
    # Gera plots e tabelas
    print(f"\n📈 GERANDO VISUALIZAÇÕES...")
    
    for metric in metrics:
        print(f"\n  📊 Processando {metric}...")
        
        try:
            # Cria plot
            plot_file = create_metric_plot(all_data, metric, output_path, args.format, args.style)
            print(f"    ✅ Plot: {plot_file}")
            
            # Cria tabelas
            csv_file, txt_file = create_summary_table(all_data, metric, output_path)
            if csv_file:
                print(f"    ✅ CSV: {csv_file}")
                print(f"    ✅ TXT: {txt_file}")
            
        except Exception as e:
            print(f"    ❌ Erro ao processar {metric}: {e}")
    
    # Relatório final
    report_file = output_path / "cli_report.txt"
    with open(report_file, 'w') as f:
        f.write("RELATÓRIO CLI - Geração de Plots e Tabelas\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Data: {datetime.now()}\n")
        f.write(f"Dados: {args.data_path}\n")
        f.write(f"Tenants: {tenants}\n")
        f.write(f"Métricas: {metrics}\n")
        f.write(f"Saída: {output_path}\n")
        f.write(f"Arquivos gerados: {list(output_path.glob('*'))}\n")
    
    print(f"\n📄 Relatório salvo: {report_file}")
    print(f"\n🎉 CLI CONCLUÍDA! Arquivos em: {output_path}")

if __name__ == "__main__":
    main()
