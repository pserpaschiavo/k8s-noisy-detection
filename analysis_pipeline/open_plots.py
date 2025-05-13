#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple script to display generated plots
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import sys

def show_plot(plot_path):
    """Show a plot from file"""
    print(f"Opening {plot_path}")
    img = mpimg.imread(plot_path)
    plt.figure(figsize=(14, 10))
    plt.imshow(img)
    plt.axis('off')  # Don't show the axes
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide path to plot file")
        sys.exit(1)
    
    plot_path = sys.argv[1]
    show_plot(plot_path)
