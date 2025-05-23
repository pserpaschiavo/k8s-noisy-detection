import matplotlib.pyplot as plt

def close_all_figures():
    """
    Closes all open matplotlib figures to prevent memory issues.
    
    This function should be called:
    1. After completing a set of related visualizations
    2. At the end of processing each phase or metric
    3. Periodically during long analysis runs
    4. Before starting a new set of visualizations
    
    Returns:
        int: The number of figures that were closed
    """
    # Get the number of open figures
    num_figs = len(plt.get_fignums())
    
    # Close all figures
    plt.close('all')
    
    return num_figs
