#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Selection Script for Kubernetes Noisy Neighbours Lab
This script helps users select and execute the appropriate pipeline version.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import textwrap

def print_header():
    """Print a styled header."""
    print("\n" + "="*80)
    print(" K8s Noisy Neighbours Lab - Analysis Pipeline Selection ".center(80, "="))
    print("="*80 + "\n")

def print_section(title):
    """Print a section header."""
    print("\n" + "-"*40)
    print(f" {title} ".center(40, "-"))
    print("-"*40)

def list_pipeline_versions():
    """List available pipeline versions."""
    print_section("Available Pipeline Versions")
    
    pipelines = [
        {
            "name": "Integrated Pipeline (Recommended)",
            "script": "main_integrated.py",
            "description": "Complete pipeline with enhanced integration between metrics, phase, and tenant analysis."
        },
        {
            "name": "Updated Pipeline",
            "script": "main_updated.py",
            "description": "Updated version with better organization but less integration."
        }
    ]
    
    for i, pipeline in enumerate(pipelines):
        print(f"\n{i+1}. {pipeline['name']}")
        print(f"   Script: {pipeline['script']}")
        description_lines = textwrap.wrap(pipeline['description'], width=75)
        for line in description_lines:
            print(f"   {line}")
    
    return pipelines

def get_pipeline_arguments(pipeline_script):
    """Get available arguments for a pipeline script."""
    try:
        result = subprocess.run(
            [sys.executable, pipeline_script, "--help"],
            capture_output=True,
            text=True,
            check=False
        )
        return result.stdout
    except Exception as e:
        return f"Error getting help: {str(e)}"

def main():
    """Main function."""
    print_header()
    
    # List available pipelines
    pipelines = list_pipeline_versions()
    
    # Ask user to select a pipeline
    print("\nSelect a pipeline to execute (or 0 to exit):")
    
    try:
        choice = int(input("\nEnter selection (1-3): "))
        if choice == 0:
            print("\nExiting...")
            return
        
        if choice < 1 or choice > len(pipelines):
            print(f"\nInvalid selection. Please choose between 1 and {len(pipelines)}.")
            return
        
        selected = pipelines[choice-1]
        print(f"\nSelected: {selected['name']} ({selected['script']})\n")
        
        # Show available arguments
        print_section("Available Arguments")
        help_text = get_pipeline_arguments(selected['script'])
        print(help_text)
        
        # Ask for arguments
        print_section("Execute Pipeline")
        print("Enter arguments for the pipeline (leave empty for defaults):")
        args = input("\nArguments: ").strip()
        
        # Confirm execution
        print(f"\nWill execute: python {selected['script']} {args}")
        confirm = input("\nProceed? (y/n): ").lower()
        
        if confirm in ['y', 'yes']:
            print("\nExecuting pipeline...\n")
            
            # Build command
            cmd = [sys.executable, selected['script']]
            if args:
                cmd.extend(args.split())
            
            # Execute
            subprocess.run(cmd)
            
            print("\nPipeline execution complete!")
        else:
            print("\nExecution cancelled.")
        
    except ValueError:
        print("\nInvalid input. Please enter a number.")
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main()
