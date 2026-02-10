"""
BasePlot - Base class for all plot types.

This module provides a foundation for different types of set theory visualizations
including Venn diagrams, UpSet plots, and Euler diagrams.
"""

import matplotlib.pyplot as plt


class BasePlot:
    """
    Base class for set theory visualizations.
    
    Provides common functionality for:
    - Figure and axis creation
    - Data loading
    - Boundary calculation
    - Coordinate axis setting
    - Style configuration
    
    Attributes:
        figsize (tuple): Figure size as (width, height) in inches
        fig (matplotlib.figure.Figure): The figure object
        ax (matplotlib.axes.Axes): The axes object
    """
    
    def __init__(self, figsize=(10, 10)):
        """
        Initialize BasePlot with figure size.
        
        Args:
            figsize (tuple): Figure size as (width, height) in inches. Default is (10, 10).
        """
        self.figsize = figsize
        self.fig = None
        self.ax = None
        
    def create_figure(self):
        """
        Create figure and axis objects.
        
        This method initializes matplotlib figure and axes with the specified size.
        """
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        
    def calculate_boundaries(self, df_edges, x_col='X', y_col='Y'):
        """
        Calculate global boundaries for coordinate axis limits.
        
        Args:
            df_edges (pd.DataFrame): DataFrame containing edge coordinate data
            x_col (str): Column name for x coordinates. Default is 'X'.
            y_col (str): Column name for y coordinates. Default is 'Y'.
            
        Returns:
            tuple: (x_min, x_max, y_min, y_max) boundary values
        """
        x_min, x_max = df_edges[x_col].min(), df_edges[x_col].max()
        y_min, y_max = df_edges[y_col].min(), df_edges[y_col].max()
        return x_min, x_max, y_min, y_max
        
    def set_axis_limits(self, x_min, x_max, y_min, y_max, padding_ratio=0.1):
        """
        Set axis limits with padding.
        
        Args:
            x_min (float): Minimum x value
            x_max (float): Maximum x value
            y_min (float): Minimum y value
            y_max (float): Maximum y value
            padding_ratio (float): Padding ratio (0.1 = 10% padding). Default is 0.1.
        """
        padding_x = (x_max - x_min) * padding_ratio
        padding_y = (y_max - y_min) * padding_ratio
        
        self.ax.set_xlim(x_min - padding_x, x_max + padding_x)
        self.ax.set_ylim(y_min - padding_y, y_max + padding_y)
        
    def set_aspect_equal(self):
        """
        Set aspect ratio to 1:1 for equal scaling.
        """
        self.ax.set_aspect('equal')
        
    def hide_axis(self):
        """
        Hide axis lines and labels.
        """
        self.ax.set_axis_off()
        
    def set_title(self, title, fontsize=14):
        """
        Set plot title.
        
        Args:
            title (str): Title text
            fontsize (int): Font size for title. Default is 14.
        """
        self.ax.set_title(title, fontsize=fontsize)
        
    def show(self):
        """
        Display the plot.
        """
        plt.show()
        
    def save(self, filename, dpi=300, bbox_inches='tight'):
        """
        Save the plot to a file.
        
        Args:
            filename (str): Output filename
            dpi (int): Dots per inch for resolution. Default is 300.
            bbox_inches (str): Bounding box in inches. Default is 'tight'.
        """
        self.fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        
    def close(self):
        """
        Close the figure to free memory.
        """
        plt.close(self.fig)
