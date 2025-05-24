#!/usr/bin/env python3
"""
CLI Simples para Geração de Plots e Tabelas - K8s Noisy Detection
"""

import sys
import os
import argparse
from pathlib import Path

# Adiciona o diretório atual ao Python path
sys.path.insert(0, '.')

def main():
    print("🎯 CLI PLOTS - K8s Noisy Detection")
    print("=" * 40)
    
    parser = argparse.ArgumentParser(description='Gerador de plots e tabelas')
    
    # Argumentos básicos
    parser.add_argument('--list-data', action='store_true', help='Lista dados disponíveis')
    parser.add_argument('--quick-test', action='store_true', help='Teste rápido')
    parser.add_argument('--tenants', '-t', help='Tenants (ex: a,b,c,d)')
    parser.add_argument('--metrics', '-m', help='Métricas (ex: memory,disk,network)')
    parser.add_argument('--output', '-o', default='cli_output', help='Diretório de saída')
    parser.add_argument('--interactive', '-i', action='store_true', help='Modo interativo')
    
    args = parser.parse_args()
    
    # Lista dados
    if args.list_data:
        print("\n📁 Dados disponíveis:")
        data_path = Path("demo-data/demo-experiment-1-round/round-1")
        if data_path.exists():
            phases = [d for d in data_path.iterdir() if d.is_dir()]
            for phase in sorted(phases):
                print(f"  📊 {phase.name}")
                tenants = [d for d in phase.iterdir() if d.is_dir()]
                for tenant in sorted(tenants):
                    metrics = [f for f in tenant.iterdir() if f.suffix == '.csv']
                    print(f"    👤 {tenant.name}: {len(metrics)} métricas")
        else:
            print("❌ Diretório de dados não encontrado")
        return
    
    # Teste rápido
    if args.quick_test:
        print("\n🚀 Executando teste rápido...")
        os.system("python3 generate_plots_tenants.py")
        return
    
    # Modo interativo
    if args.interactive:
        print("\n🎯 MODO INTERATIVO")
        print("=" * 30)
        
        print("\nTenants disponíveis:")
        print("1. tenant-a")
        print("2. tenant-b") 
        print("3. tenant-c")
        print("4. tenant-d")
        print("5. Todos")
        
        choice = input("\nEscolha (1-5): ").strip()
        
        if choice == "5":
            tenants = "a,b,c,d"
        else:
            tenant_map = {"1": "a", "2": "b", "3": "c", "4": "d"}
            tenants = tenant_map.get(choice, "a")
        
        print("\nMétricas disponíveis:")
        print("1. memory_usage")
        print("2. disk_throughput_total")
        print("3. network_total_bandwidth") 
        print("4. cpu_usage")
        print("5. Todas")
        
        metric_choice = input("\nEscolha (1-5): ").strip()
        
        if metric_choice == "5":
            metrics = "memory,disk,network,cpu"
        else:
            metric_map = {"1": "memory", "2": "disk", "3": "network", "4": "cpu"}
            metrics = metric_map.get(metric_choice, "memory")
        
        # Executa com parâmetros escolhidos
        args.tenants = tenants
        args.metrics = metrics
    
    # Processamento principal
    if args.tenants and args.metrics:
        print(f"\n📊 Processando:")
        print(f"  👥 Tenants: {args.tenants}")
        print(f"  📊 Métricas: {args.metrics}")
        print(f"  📁 Saída: {args.output}")
        
        # Cria comando para executar
        tenants_list = [f"tenant-{t}" if not t.startswith('tenant-') else t for t in args.tenants.split(',')]
        metrics_full = {
            'memory': 'memory_usage',
            'disk': 'disk_throughput_total', 
            'network': 'network_total_bandwidth',
            'cpu': 'cpu_usage'
        }
        
        print(f"\n🔧 Executando geração de plots...")
        
        # Importa e executa o gerador principal
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            
            # Cria diretório de saída
            output_path = Path(args.output)
            output_path.mkdir(exist_ok=True)
            
            print(f"✅ Bibliotecas importadas")
            print(f"✅ Diretório criado: {output_path}")
            
            # Chama o script principal de geração
            cmd = f"python3 generate_plots_tenants.py"
            print(f"🚀 Executando: {cmd}")
            os.system(cmd)
            
        except ImportError as e:
            print(f"❌ Erro de importação: {e}")
            print("💡 Tentando executar script externo...")
            os.system("python3 generate_plots_tenants.py")
        
    else:
        print("\n💡 Use --help para ver opções disponíveis")
        print("💡 Exemplos:")
        print("  python3 cli_plots_simple.py --list-data")
        print("  python3 cli_plots_simple.py --quick-test") 
        print("  python3 cli_plots_simple.py --interactive")
        print("  python3 cli_plots_simple.py --tenants a,b --metrics memory,network")

if __name__ == "__main__":
    main()
