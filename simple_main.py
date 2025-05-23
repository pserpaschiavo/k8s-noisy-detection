#!/usr/bin/env python3
"""
Versão simplificada do script principal para testar execução básica
"""
import argparse
import sys
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="K8s Noisy Neighbor Detection")
    parser.add_argument("--data-dir", required=True, help="Diretório de dados")
    parser.add_argument("--output-dir", required=True, help="Diretório de saída")
    parser.add_argument("--run-root-cause", action="store_true", help="Executar análise de causa raiz")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    print(f"Script executado com sucesso!")
    print(f"Diretório de dados: {args.data_dir}")
    print(f"Diretório de saída: {args.output_dir}")
    print(f"Análise de causa raiz: {'Sim' if args.run_root_cause else 'Não'}")

if __name__ == "__main__":
    main()
