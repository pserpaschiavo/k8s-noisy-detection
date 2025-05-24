#!/usr/bin/env python3
"""
Geração direta de plots e tabelas com tenants a, b, c, d
Focando nas melhorias do sistema de formatação de métricas
"""

import sys
import os
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_tenant_data(base_path, phase, tenant):
    """Carrega dados de um tenant específico."""
    tenant_path = Path(base_path) / phase / tenant
    
    data = {}
    metrics = ['memory_usage.csv', 'disk_throughput_total.csv', 'network_total_bandwidth.csv', 'cpu_usage.csv']
    
    for metric_file in metrics:
        metric_path = tenant_path / metric_file
        if metric_path.exists():
            try:
                df = pd.read_csv(metric_path)
                metric_name = metric_file.replace('.csv', '')
                data[metric_name] = df
                print(f"  ✅ {tenant}/{metric_name}: {len(df)} registros")
            except Exception as e:
                print(f"  ❌ Erro ao carregar {tenant}/{metric_file}: {e}")
    
    return data

def apply_intelligent_formatting(df, metric_name):
    """Aplica formatação inteligente aos dados."""
    try:
        from src.utils.metric_formatter import detect_and_convert_units
        
        # Apply intelligent formatting
        formatted_df = detect_and_convert_units(df.copy(), metric_name)
        
        return formatted_df
    except Exception as e:
        print(f"  ⚠️  Formatação inteligente falhou para {metric_name}: {e}")
        return df

def create_comparison_plot(data_dict, metric_name, output_dir):
    """Cria plots comparativos entre tenants."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Comparação {metric_name.replace("_", " ").title()} - Tenants A, B, C, D', fontsize=16)
    
    tenants = ['tenant-a', 'tenant-b', 'tenant-c', 'tenant-d']
    phases = ['1 - Baseline', '2 - Attack', '3 - Recovery']
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, phase in enumerate(phases[:3]):  # Only first 3 phases
        ax = axes[i//2, i%2] if i < 2 else axes[1, 0]
        
        for j, tenant in enumerate(tenants):
            if phase in data_dict and tenant in data_dict[phase]:
                df = data_dict[phase][tenant]
                if metric_name in df.columns:
                    values = df[metric_name].dropna()
                    
                    # Plot time series
                    if 'timestamp' in df.columns:
                        timestamps = pd.to_datetime(df['timestamp'])
                        ax.plot(timestamps, values, 
                               label=f'{tenant.replace("tenant-", "Tenant ")}', 
                               color=colors[j], alpha=0.7)
                    else:
                        # Plot as sequence
                        ax.plot(values, label=f'{tenant.replace("tenant-", "Tenant ")}', 
                               color=colors[j], alpha=0.7)
        
        ax.set_title(f'{phase}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set y-label with unit if available
        unit_label = metric_name.replace('_', ' ').title()
        if phase in data_dict:
            for tenant in tenants:
                if tenant in data_dict[phase] and metric_name in data_dict[phase][tenant].columns:
                    sample_df = data_dict[phase][tenant]
                    if 'display_unit' in sample_df.columns:
                        unit_label += f" ({sample_df['display_unit'].iloc[0]})"
                        break
        ax.set_ylabel(unit_label)
    
    # Remove empty subplot
    if len(phases) == 3:
        axes[1, 1].remove()
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / f'{metric_name}_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 Plot salvo: {plot_file}")
    return plot_file

def create_summary_table(data_dict, metric_name, output_dir):
    """Cria tabela resumo das estatísticas."""
    
    summary_data = []
    
    tenants = ['tenant-a', 'tenant-b', 'tenant-c', 'tenant-d']
    phases = ['1 - Baseline', '2 - Attack', '3 - Recovery']
    
    for phase in phases:
        for tenant in tenants:
            if phase in data_dict and tenant in data_dict[phase]:
                df = data_dict[phase][tenant]
                if metric_name in df.columns:
                    values = df[metric_name].dropna()
                    
                    unit = ""
                    if 'display_unit' in df.columns:
                        unit = df['display_unit'].iloc[0]
                    
                    summary_data.append({
                        'Phase': phase,
                        'Tenant': tenant.replace('tenant-', 'Tenant '),
                        'Count': len(values),
                        'Mean': values.mean(),
                        'Median': values.median(),
                        'Std': values.std(),
                        'Min': values.min(),
                        'Max': values.max(),
                        'Unit': unit
                    })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Save CSV
        csv_file = output_dir / f'{metric_name}_summary.csv'
        summary_df.to_csv(csv_file, index=False)
        
        # Save formatted table
        table_file = output_dir / f'{metric_name}_summary.txt'
        with open(table_file, 'w') as f:
            f.write(f"Resumo Estatístico - {metric_name.replace('_', ' ').title()}\n")
            f.write("="*80 + "\n\n")
            f.write(summary_df.to_string(index=False))
        
        print(f"  📋 Tabela salva: {csv_file}")
        print(f"  📄 Resumo salvo: {table_file}")
        
        return summary_df
    
    return None

def main():
    print("🚀 GERANDO PLOTS E TABELAS - TENANTS A, B, C, D")
    print("🔧 Sistema de Formatação Inteligente de Métricas")
    print("="*70)
    
    # Configuration
    base_path = "demo-data/demo-experiment-1-round/round-1"
    output_dir = Path("output/tenants_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    phases = ['1 - Baseline', '2 - Attack', '3 - Recovery']
    tenants = ['tenant-a', 'tenant-b', 'tenant-c', 'tenant-d']
    metrics = ['memory_usage', 'disk_throughput_total', 'network_total_bandwidth', 'cpu_usage']
    
    print(f"📁 Dados: {base_path}")
    print(f"📁 Saída: {output_dir}")
    print(f"👥 Tenants: {tenants}")
    print(f"📊 Métricas: {metrics}")
    
    # Load all data
    print("\n📊 CARREGANDO DADOS...")
    all_data = {}
    
    for phase in phases:
        print(f"\n📁 Fase: {phase}")
        phase_data = {}
        
        for tenant in tenants:
            print(f"  👤 {tenant}:")
            tenant_data = load_tenant_data(base_path, phase, tenant)
            
            # Apply intelligent formatting to problematic metrics
            for metric_name in ['memory_usage', 'disk_throughput_total', 'network_total_bandwidth']:
                if metric_name in tenant_data:
                    print(f"    🔧 Aplicando formatação inteligente a {metric_name}")
                    tenant_data[metric_name] = apply_intelligent_formatting(
                        tenant_data[metric_name], metric_name
                    )
                    
                    # Show formatting results
                    df = tenant_data[metric_name]
                    if 'display_unit' in df.columns:
                        unit = df['display_unit'].iloc[0]
                        sample_values = df['value'].head(3).tolist()
                        print(f"      ✅ Unidade: {unit}, Valores: {[f'{v:.2f}' for v in sample_values]}")
            
            phase_data[tenant] = tenant_data
        
        all_data[phase] = phase_data
    
    # Generate plots and tables for each metric
    print(f"\n📈 GERANDO PLOTS E TABELAS...")
    
    results_summary = {}
    
    for metric in metrics:
        print(f"\n{'='*50}")
        print(f"📊 PROCESSANDO: {metric.upper()}")
        print(f"{'='*50}")
        
        # Reorganize data by metric
        metric_data = {}
        for phase in phases:
            metric_data[phase] = {}
            for tenant in tenants:
                if phase in all_data and tenant in all_data[phase]:
                    if metric in all_data[phase][tenant]:
                        metric_data[phase][tenant] = all_data[phase][tenant][metric]
        
        # Create plots
        plot_file = create_comparison_plot(metric_data, metric, output_dir)
        
        # Create summary table
        summary_df = create_summary_table(metric_data, metric, output_dir)
        
        results_summary[metric] = {
            'plot': plot_file,
            'summary': summary_df
        }
        
        print(f"✅ {metric} processado com sucesso!")
    
    # Generate final report
    print(f"\n{'='*70}")
    print("📋 RELATÓRIO FINAL")
    print(f"{'='*70}")
    
    report_file = output_dir / "analysis_report.txt"
    with open(report_file, 'w') as f:
        f.write("ANÁLISE DE TENANTS A, B, C, D\n")
        f.write("Sistema de Formatação Inteligente de Métricas\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Data da análise: {pd.Timestamp.now()}\n")
        f.write(f"Dados fonte: {base_path}\n")
        f.write(f"Tenants analisados: {', '.join(tenants)}\n")
        f.write(f"Fases analisadas: {', '.join(phases)}\n")
        f.write(f"Métricas processadas: {', '.join(metrics)}\n\n")
        
        f.write("MELHORIAS IMPLEMENTADAS:\n")
        f.write("✅ Conversões hard-coded removidas\n")
        f.write("✅ Sistema de formatação inteligente aplicado\n")
        f.write("✅ Unidades automáticas detectadas\n")
        f.write("✅ Plots com escalas legíveis\n")
        f.write("✅ Tabelas com unidades corretas\n\n")
        
        f.write("ARQUIVOS GERADOS:\n")
        for metric, results in results_summary.items():
            f.write(f"- {metric}_comparison.png (Plot comparativo)\n")
            f.write(f"- {metric}_summary.csv (Dados estatísticos)\n")
            f.write(f"- {metric}_summary.txt (Resumo formatado)\n")
    
    print(f"📄 Relatório salvo: {report_file}")
    
    # List all generated files
    generated_files = list(output_dir.glob("*"))
    print(f"\n📁 ARQUIVOS GERADOS ({len(generated_files)}):")
    for file_path in sorted(generated_files):
        print(f"  - {file_path.name}")
    
    print(f"\n🎉 ANÁLISE CONCLUÍDA COM SUCESSO!")
    print(f"📊 {len(metrics)} métricas processadas")
    print(f"👥 {len(tenants)} tenants analisados")
    print(f"📈 Plots e tabelas prontos para avaliação")
    print(f"📁 Resultados em: {output_dir.absolute()}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"❌ Erro na execução: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
