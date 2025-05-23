#!/usr/bin/env python3
"""
Script para verificar erros de sintaxe nos módulos
"""
import os
import sys

# Diretório raiz do projeto
project_dir = os.path.abspath(os.path.dirname(__file__))
modules_dir = os.path.join(project_dir, 'analysis_modules')

with open("syntax_check_results.txt", "w") as f:
    f.write("=== INICIANDO VERIFICAÇÃO DE SINTAXE ===\n")

    # Testar cada arquivo Python no diretório analysis_modules
    for filename in os.listdir(modules_dir):
        if filename.endswith(".py"):
            filepath = os.path.join(modules_dir, filename)
            try:
                with open(filepath, "r") as module_file:
                    code = module_file.read()
                    compile(code, filepath, 'exec')
                    f.write(f"✓ {filename}: Sintaxe OK\n")
            except SyntaxError as e:
                f.write(f"✗ {filename}: ERRO DE SINTAXE - Linha {e.lineno}, Coluna {e.offset}\n")
                f.write(f"  {e.text}\n")
                f.write(f"  {' ' * (e.offset - 1)}^\n")
                f.write(f"  {str(e)}\n\n")
            except Exception as e:
                f.write(f"✗ {filename}: ERRO - {str(e)}\n\n")

    f.write("\n=== VERIFICAÇÃO CONCLUÍDA ===\n")
