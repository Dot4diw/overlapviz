"""
Unified Plot Types Module.

This module provides a unified interface for creating different types of
set theory visualizations: Venn diagrams, UpSet plots, and Euler diagrams.

All plot classes inherit from BasePlot, providing a consistent API for
common operations like figure creation, styling, and export.
"""

from base_plot import BasePlot
from venn_plot import VennPlot
from upset_plot import UpSetPlot
from euler_plot import EulerPlot


__all__ = ['BasePlot', 'VennPlot', 'UpSetPlot', 'EulerPlot']


# Factory function for convenient plot creation
def create_plot(plot_type, **kwargs):
    """
    Factory function to create plot instances.
    
    Args:
        plot_type (str): Type of plot to create ('venn', 'upset', or 'euler')
        **kwargs: Additional arguments to pass to the plot class constructor
        
    Returns:
        BasePlot subclass: Instance of the requested plot type
        
    Raises:
        ValueError: If plot_type is not recognized
    """
    plot_classes = {
        'venn': VennPlot,
        'upset': UpSetPlot,
        'euler': EulerPlot
    }
    
    plot_type_lower = plot_type.lower()
    if plot_type_lower not in plot_classes:
        raise ValueError(
            f"Unknown plot type: {plot_type}. "
            f"Available types: {list(plot_classes.keys())}"
        )
    
    return plot_classes[plot_type_lower](**kwargs)
