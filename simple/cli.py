#!/usr/bin/env python3
"""
K8s Noisy Detection - Simplified CLI Interface
==============================================

Interface CLI unificada para o sistema simplificado de análise K8s.
Substitui o pipeline complexo por uma abordagem modular e clara.

Autor: Phil
Versão: 2.0 (Simplificado)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from core import SimpleK8sAnalyzer

# Import extended and advanced analyzers with error handling
try:
    from extended import ExtendedK8sAnalyzer
    EXTENDED_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Extended analyzer not available: {e}")
    ExtendedK8sAnalyzer = SimpleK8sAnalyzer  # Fallback
    EXTENDED_AVAILABLE = False

try:
    from advanced import AdvancedK8sAnalyzer
    ADVANCED_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Advanced analyzer not available: {e}")
    AdvancedK8sAnalyzer = SimpleK8sAnalyzer  # Fallback
    ADVANCED_AVAILABLE = False

def setup_argparse():
    """Configura argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Sistema simplificado de análise K8s com configuração YAML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Análise básica com configuração padrão
  python cli.py --mode basic --data demo-data/demo-experiment-1-round/round-1

  # Análise com configuração personalizada
  python cli.py --config config/extended_config.yaml

  # Análise interativa
  python cli.py --interactive

  # Debug de problemas nos dados
  python cli.py --debug-only --data demo-data/demo-experiment-1-round/round-1

Modos disponíveis:
  - basic: Análise rápida (< 5 min) com estatísticas essenciais
  - extended: Análise intermediária (< 15 min) com técnicas avançadas
  - advanced: Análise completa (30+ min) com ML e análise de sinais
        """
    )
    
    # Configuração
    config_group = parser.add_argument_group('Configuração')
    config_group.add_argument(
        '--config', '-c',
        type=str,
        help='Caminho para arquivo YAML de configuração'
    )
    config_group.add_argument(
        '--mode', '-m',
        choices=['basic', 'extended', 'advanced'],
        default='basic',
        help='Modo de análise (padrão: basic)'
    )
    
    # Dados
    data_group = parser.add_argument_group('Dados')
    data_group.add_argument(
        '--data', '-d',
        type=str,
        help='Caminho para diretório de dados'
    )
    data_group.add_argument(
        '--tenants', '-t',
        nargs='+',
        default=['a', 'b', 'c', 'd'],
        help='Lista de tenants para analisar (padrão: a b c d)'
    )
    data_group.add_argument(
        '--metrics',
        nargs='+',
        default=['memory_usage', 'disk_throughput_total', 'network_total_bandwidth', 'cpu_usage'],
        help='Métricas para analisar'
    )
    
    # Output
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        '--output', '-o',
        type=str,
        default='./output/simple_analysis',
        help='Diretório de output (padrão: ./output/simple_analysis)'
    )
    output_group.add_argument(
        '--format',
        choices=['png', 'svg', 'pdf'],
        default='png',
        help='Formato dos plots (padrão: png)'
    )
    output_group.add_argument(
        '--no-plots',
        action='store_true',
        help='Não gerar plots (apenas estatísticas)'
    )
    
    # Controle de execução
    exec_group = parser.add_argument_group('Execução')
    exec_group.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Modo interativo'
    )
    exec_group.add_argument(
        '--debug-only',
        action='store_true',
        help='Apenas debug de problemas nos dados'
    )
    exec_group.add_argument(
        '--quick',
        action='store_true',
        help='Análise rápida completa (equivale ao modo basic)'
    )
    exec_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Output verboso'
    )
    
    return parser

def interactive_mode():
    """Executa modo interativo."""
    print("\n🎯 MODO INTERATIVO - Sistema Simplificado K8s")
    print("=" * 60)
    
    # Solicita configurações
    data_path = input("📁 Caminho para dados [demo-data/demo-experiment-1-round/round-1]: ").strip()
    if not data_path:
        data_path = "demo-data/demo-experiment-1-round/round-1"
    
    mode = input("🔧 Modo de análise (basic/extended/advanced) [basic]: ").strip()
    if not mode or mode not in ['basic', 'extended', 'advanced']:
        mode = 'basic'
    
    output_dir = input("📊 Diretório de output [./output/interactive_analysis]: ").strip()
    if not output_dir:
        output_dir = "./output/interactive_analysis"
    
    tenants = input("👥 Tenants (separados por espaço) [a b c d]: ").strip()
    if not tenants:
        tenants = ['a', 'b', 'c', 'd']
    else:
        tenants = tenants.split()
    
    print(f"\n🚀 Iniciando análise interativa ({mode.upper()})...")
    
    # Cria configuração dinâmica
    config = {
        'analysis': {
            'mode': mode,
            'tenants': tenants,
            'metrics': ['memory_usage', 'disk_throughput_total', 'network_total_bandwidth', 'cpu_usage']
        },
        'data': {'path': data_path},
        'output': {'directory': output_dir, 'format': 'png'},
        'debug': {'verbose': True},
        'plots': {'basic_timeseries': True, 'boxplots': True, 'figsize': [12, 8], 'dpi': 300},
        'processing': {'auto_format': True}
    }
    
    try:
        # Seleciona analyzer baseado no modo
        if mode == 'extended':
            config_path = 'config/extended_config.yaml'
            if Path(config_path).exists():
                analyzer = ExtendedK8sAnalyzer(config_path)
                analyzer.data_path = data_path
                result = analyzer.extended_analysis()
            else:
                print("⚠️ Configuração extended não encontrada, usando análise básica")
                analyzer = SimpleK8sAnalyzer(data_path=data_path)
                analyzer.config.update(config)
                analyzer.output_dir = Path(output_dir)
                analyzer.output_dir.mkdir(parents=True, exist_ok=True)
                result = analyzer.quick_analysis()
        elif mode == 'advanced':
            config_path = 'config/advanced_config.yaml'
            if Path(config_path).exists():
                analyzer = AdvancedK8sAnalyzer(config_path)
                analyzer.data_path = data_path
                result = analyzer.run_advanced_analysis()
            else:
                print("⚠️ Configuração advanced não encontrada, usando análise básica")
                analyzer = SimpleK8sAnalyzer(data_path=data_path)
                analyzer.config.update(config)
                analyzer.output_dir = Path(output_dir)
                analyzer.output_dir.mkdir(parents=True, exist_ok=True)
                result = analyzer.quick_analysis()
        else:  # basic
            analyzer = SimpleK8sAnalyzer(data_path=data_path)
            analyzer.config.update(config)
            analyzer.output_dir = Path(output_dir)
            analyzer.output_dir.mkdir(parents=True, exist_ok=True)
            result = analyzer.quick_analysis()
        
        if result:
            print(f"\n✅ Análise interativa ({mode.upper()}) concluída!")
            if hasattr(analyzer, 'output_dir'):
                print(f"📂 Resultados salvos em: {analyzer.output_dir}")
            elif 'output_dir' in result:
                print(f"📂 Resultados salvos em: {result['output_dir']}")
            
            # Pergunta se quer ver resumo
            show_summary = input("\n📄 Mostrar resumo dos dados? (s/n) [s]: ").strip().lower()
            if show_summary != 'n':
                if hasattr(analyzer, 'show_data_summary'):
                    analyzer.show_data_summary()
        else:
            print(f"\n❌ Análise {mode.upper()} falhou")
            
    except Exception as e:
        print(f"\n❌ Erro na análise: {e}")
        return False
    
    return True

def debug_only_mode(data_path: str, tenants: list, verbose: bool):
    """Executa apenas debug de problemas nos dados."""
    print("\n🐛 MODO DEBUG - Verificação de Problemas nos Dados")
    print("=" * 60)
    
    try:
        config = {
            'analysis': {'tenants': tenants, 'metrics': ['memory_usage', 'disk_throughput_total', 'network_total_bandwidth', 'cpu_usage']},
            'data': {'path': data_path},
            'output': {'directory': './output/debug_only'},
            'debug': {'verbose': verbose},
            'processing': {'auto_format': False}
        }
        
        analyzer = SimpleK8sAnalyzer(data_path=data_path)
        analyzer.config.update(config)
        
        # Apenas carrega dados e faz debug
        print("📥 Carregando dados...")
        success = analyzer.load_data()
        
        if success:
            print("\n🔍 Analisando problemas...")
            issues = analyzer.debug_data_issues()
            
            print("\n📊 Resumo dos dados:")
            analyzer.show_data_summary()
            
            if issues:
                print(f"\n⚠️ {len(issues)} problemas encontrados")
                print("📄 Relatório detalhado salvo em: output/debug_only/data_issues_report.txt")
            else:
                print("\n✅ Nenhum problema detectado nos dados!")
        else:
            print("\n❌ Falha no carregamento dos dados")
            
    except Exception as e:
        print(f"\n❌ Erro no debug: {e}")
        return False
    
    return True

def main():
    """Função principal."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Modo interativo
    if args.interactive:
        return interactive_mode()
    
    # Modo debug apenas
    if args.debug_only:
        if not args.data:
            print("❌ Erro: --data é obrigatório no modo --debug-only")
            return False
        return debug_only_mode(args.data, args.tenants, args.verbose)
    
    # Análise normal
    try:
        # Determina configuração e analyzer
        analyzer_class = SimpleK8sAnalyzer
        
        if args.config:
            if not Path(args.config).exists():
                print(f"❌ Arquivo de configuração não encontrado: {args.config}")
                return False
            
            # Determina tipo de analyzer baseado no modo no arquivo config
            if 'extended' in args.config:
                analyzer_class = ExtendedK8sAnalyzer
            elif 'advanced' in args.config:
                analyzer_class = AdvancedK8sAnalyzer
                
            analyzer = analyzer_class(args.config)
            if args.data:
                analyzer.data_path = args.data
        else:
            # Determina analyzer baseado no modo
            if args.mode == 'extended' and EXTENDED_AVAILABLE:
                analyzer_class = ExtendedK8sAnalyzer
                config_path = 'config/extended_config.yaml'
            elif args.mode == 'advanced' and ADVANCED_AVAILABLE:
                analyzer_class = AdvancedK8sAnalyzer
                config_path = 'config/advanced_config.yaml'
            else:
                analyzer_class = SimpleK8sAnalyzer
                config_path = 'config/basic_config.yaml'
                if args.mode != 'basic':
                    print(f"⚠️ Modo {args.mode} não disponível, usando modo básico")
            
            # Tenta usar arquivo de configuração padrão
            if Path(config_path).exists():
                analyzer = analyzer_class(config_path)
                if args.data:
                    analyzer.data_path = args.data
            else:
                # Fallback para configuração manual (apenas para basic)
                if args.mode != 'basic':
                    print(f"❌ Erro: Arquivo {config_path} não encontrado para modo {args.mode}")
                    print("⚠️ Executando análise básica como fallback")
                    analyzer_class = SimpleK8sAnalyzer
                
                if not args.data:
                    print("❌ Erro: --data é obrigatório quando não há arquivo de configuração")
                    return False
                    
                analyzer = analyzer_class(data_path=args.data)
                
                # Atualiza configuração com argumentos
                analyzer.config['analysis']['tenants'] = args.tenants
                analyzer.config['analysis']['metrics'] = args.metrics
                analyzer.config['analysis']['mode'] = args.mode
                analyzer.config['output']['directory'] = args.output
                analyzer.config['output']['format'] = args.format
                analyzer.config['debug']['verbose'] = args.verbose
                analyzer.output_dir = Path(args.output)
                analyzer.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🚀 Iniciando análise {args.mode.upper()}...")
        print(f"📁 Dados: {analyzer.data_path}")
        print(f"📊 Output: {analyzer.output_dir}")
        
        # Executa análise
        if args.quick or args.mode == 'basic':
            result = analyzer.quick_analysis()
        elif args.mode == 'extended':
            print("🔧 Executando análise EXTENDED com técnicas avançadas...")
            if EXTENDED_AVAILABLE and isinstance(analyzer, ExtendedK8sAnalyzer):
                result = analyzer.extended_analysis()
            else:
                print("⚠️ Fallback para análise básica (Extended não disponível)")
                result = analyzer.quick_analysis()
        elif args.mode == 'advanced':
            print("🚀 Executando análise ADVANCED com ML e processamento de sinais...")
            print("⏰ Tempo estimado: 30+ minutos")
            confirm = input("Continuar? (s/n) [s]: ").strip().lower()
            if confirm == 'n':
                print("❌ Análise cancelada pelo usuário")
                return False
            
            if ADVANCED_AVAILABLE and isinstance(analyzer, AdvancedK8sAnalyzer):
                result = analyzer.run_advanced_analysis()
            else:
                print("⚠️ Fallback para análise básica (Advanced não disponível)")
                result = analyzer.quick_analysis()
        else:
            result = analyzer.quick_analysis()
        
        if result:
            print(f"\n🎉 Análise {args.mode.upper()} concluída com sucesso!")
            print(f"📂 Resultados em: {result['output_dir']}")
            
            if args.verbose:
                print("\n📄 Arquivos gerados:")
                output_path = Path(result['output_dir'])
                for file in output_path.glob("*"):
                    if file.is_file():
                        print(f"  - {file.name}")
                for subdir in output_path.glob("*/"):
                    files = list(subdir.glob("*"))
                    if files:
                        print(f"  - {subdir.name}/: {len(files)} arquivos")
        else:
            print(f"\n❌ Análise {args.mode.upper()} falhou")
            return False
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Análise interrompida pelo usuário")
        return False
    except Exception as e:
        print(f"\n❌ Erro na análise: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
