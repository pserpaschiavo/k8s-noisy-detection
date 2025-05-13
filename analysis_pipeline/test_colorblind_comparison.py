#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare standard visualizations with colorblind-friendly visualizations
Generates the same plots with both modes for comparison.
"""

import os
import sys
import logging
import argparse
import subprocess
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

def generate_plots(colorblind_mode=True):
    """Generate tenant comparison plots with or without colorblind mode"""
    mode = "colorblind" if colorblind_mode else "standard"
    output_dir = f"/home/phil/Projects/k8s-noisy-lab-data-pipe/analysis_pipeline/results/comparison_{mode}"
    
    cmd = [
        "python", "tenant_comparison_module.py",
        "--experiment", "2025-05-11/16-58-00/default-experiment-1",
        "--output", output_dir
    ]
    
    if colorblind_mode:
        cmd.extend(["--colorblind-friendly", "True"])
    else:
        cmd.extend(["--colorblind-friendly", "False"])
    
    logging.info(f"Generating {mode} plots with command: {' '.join(cmd)}")
    result = subprocess.run(cmd, 
                          capture_output=True, 
                          text=True,
                          cwd=os.path.dirname(os.path.abspath(__file__)))
    
    if result.returncode != 0:
        logging.error(f"Failed to generate {mode} plots")
        logging.error(f"Error: {result.stderr}")
        return None
    
    return os.path.join(output_dir, "tenant_comparison")

def compare_plots(standard_dir, colorblind_dir):
    """Compare the two sets of plots side by side"""
    logging.info("Comparing plots side by side")
    
    # Find all PNG files in both directories
    standard_plots = sorted(list(Path(standard_dir).glob("*.png")))
    colorblind_plots = sorted(list(Path(colorblind_dir).glob("*.png")))
    
    if not standard_plots or not colorblind_plots:
        logging.error("No plots found in one or both directories")
        return
    
    # Create a figure for each pair of plots
    for std_plot, cb_plot in zip(standard_plots, colorblind_plots):
        if std_plot.name != cb_plot.name:
            logging.warning(f"Plot names don't match: {std_plot.name} vs {cb_plot.name}")
            continue
        
        plt.figure(figsize=(20, 10))
        
        # Standard plot
        plt.subplot(1, 2, 1)
        img = mpimg.imread(std_plot)
        plt.imshow(img)
        plt.title(f"Standard Colors: {std_plot.stem}")
        plt.axis('off')
        
        # Colorblind-friendly plot
        plt.subplot(1, 2, 2)
        img = mpimg.imread(cb_plot)
        plt.imshow(img)
        plt.title(f"Colorblind-Friendly: {cb_plot.stem}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.suptitle(f"Comparison: {std_plot.stem}", fontsize=16, y=0.98)
        
        # Save and show the comparison
        comparison_dir = Path("/home/phil/Projects/k8s-noisy-lab-data-pipe/analysis_pipeline/results/colorblind_comparison")
        comparison_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(comparison_dir / f"comparison_{std_plot.stem}.png", dpi=300, bbox_inches='tight')
        logging.info(f"Saved comparison for {std_plot.stem}")
    
    plt.close('all')
    logging.info(f"All comparisons saved to {comparison_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare standard and colorblind-friendly visualizations')
    parser.add_argument('--show', action='store_true', help='Show the plots instead of just saving them')
    args = parser.parse_args()
    
    # Generate both sets of plots
    logging.info("Generating standard visualizations...")
    standard_dir = generate_plots(colorblind_mode=False)
    
    logging.info("Generating colorblind-friendly visualizations...")
    colorblind_dir = generate_plots(colorblind_mode=True)
    
    if standard_dir and colorblind_dir:
        # Compare the plots
        compare_plots(standard_dir, colorblind_dir)
        
        # Open plots if requested
        if args.show:
            comparison_dir = "/home/phil/Projects/k8s-noisy-lab-data-pipe/analysis_pipeline/results/colorblind_comparison"
            subprocess.run(["xdg-open", comparison_dir], check=False)
        
        logging.info("Comparison complete!")
    else:
        logging.error("Failed to generate plots for comparison")
        sys.exit(1)
