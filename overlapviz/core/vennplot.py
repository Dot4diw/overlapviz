"""
VennPlot - Class for drawing Venn diagrams.

This module provides functionality to create polygon-based Venn diagrams
with customizable styling, labels, and overlap data integration.
"""

import pickle
import pandas as pd
import matplotlib.cm as cm
from matplotlib.patches import Polygon
from matplotlib.colors import to_rgba
from base import BasePlot


class VennPlot(BasePlot):
    """
    Class for creating polygon-based Venn diagrams.
    
    This class extends BasePlot to provide Venn diagram specific functionality
    including geometric data loading, overlap data integration, and 
    multi-layer rendering (fill, borders, labels).
    
    Attributes:
        geometric_data (dict): Dictionary containing geometric shape data
        overlap_data (pd.DataFrame): DataFrame containing overlap information
        df_edges (pd.DataFrame): Edge coordinate data
        set_label (pd.DataFrame): Set label coordinates and IDs
        df_label (pd.DataFrame): Region label coordinates with overlap sizes
        zorder_fill (int): Z-order for filled regions
        zorder_border (int): Z-order for borders
        border_linewidth (int): Line width for borders
        border_color (str): Color for borders
        fill_alpha (float): Transparency for filled regions (0-1)
        colormap (str): Matplotlib colormap name for fill colors
    """
    
    def __init__(self, figsize=(10, 10), **kwargs):
        """
        Initialize VennPlot.
        
        Args:
            figsize (tuple): Figure size as (width, height) in inches. Default is (10, 10).
            **kwargs: Additional keyword arguments for styling:
                - zorder_fill (int): Z-order for filled regions. Default is 1.
                - zorder_border (int): Z-order for borders. Default is 2.
                - border_linewidth (int): Border line width. Default is 2.
                - border_color (str): Border color. Default is 'black'.
                - fill_alpha (float): Fill transparency (0-1). Default is 0.4.
                - colormap (str): Colormap name. Default is 'viridis'.
        """
        super().__init__(figsize)
        
        # Styling parameters with defaults
        self.zorder_fill = kwargs.get('zorder_fill', 1)
        self.zorder_border = kwargs.get('zorder_border', 2)
        self.border_linewidth = kwargs.get('border_linewidth', 2)
        self.border_color = kwargs.get('border_color', 'black')
        self.fill_alpha = kwargs.get('fill_alpha', 0.4)
        self.colormap = kwargs.get('colormap', 'viridis')
        
        # Data containers
        self.geometric_data = None
        self.overlap_data = None
        self.df_edges = None
        self.set_label = None
        self.df_label = None
        
    def load_geometric_data(self, pickle_file, shape_key='shape404'):
        """
        Load geometric data from pickle file.
        
        Args:
            pickle_file (str): Path to pickle file containing geometric data
            shape_key (str): Key in dictionary to access shape data. Default is 'shape404'.
            
        Raises:
            FileNotFoundError: If pickle file does not exist
            KeyError: If shape_key not found in loaded data
        """
        with open(pickle_file, 'rb') as f:
            self.geometric_data = pickle.load(f)
        
        # Extract data components
        self.df_edges = self.geometric_data[shape_key]['set_edge']
        self.set_label = self.geometric_data[shape_key]['set_label']
        self.df_label = self.geometric_data[shape_key]['region_label']
        
    def load_overlap_data(self, csv_file):
        """
        Load overlap data from CSV file and merge with region labels.
        
        This method reads overlap data, replaces '&' with '_' in set names,
        and merges with region label data based on matching IDs.
        
        Args:
            csv_file (str): Path to CSV file containing overlap data
            
        Raises:
            FileNotFoundError: If CSV file does not exist
        """
        self.overlap_data = pd.read_csv(csv_file)
        
        # Replace '&' with '_' in set_names for matching
        self.overlap_data['set_names'] = self.overlap_data['set_names'].str.replace(' & ', '_')
        
        # Merge overlap data with region labels
        self.df_label = pd.merge(
            self.df_label, 
            self.overlap_data, 
            left_on='id', 
            right_on='set_names', 
            how='left'
        )
        
    def _generate_colors(self):
        """
        Generate colors for each unique region ID using colormap.
        
        Returns:
            tuple: (unique_ids_list, id_to_index_dict, colors_list)
        """
        unique_ids = self.df_edges['id'].unique()
        id_to_index = {id_val: i for i, id_val in enumerate(unique_ids)}
        colors = [cm.get_cmap(self.colormap)(i / len(unique_ids)) for i in range(len(unique_ids))]
        
        return unique_ids, id_to_index, colors
        
    def draw_filled_regions(self):
        """
        Draw filled regions for all Venn diagram areas (without borders).
        
        This method renders all regions with semi-transparent fills using
        the configured colormap. Borders are rendered separately in a
        subsequent layer to ensure clean edges.
        """
        unique_ids, id_to_index, colors = self._generate_colors()
        
        for name, group in self.df_edges.groupby('id'):
            verts = group[['X', 'Y']].values
            index = id_to_index[name]
            
            poly = Polygon(
                verts,
                closed=True,
                facecolor=to_rgba(colors[index], self.fill_alpha),
                edgecolor='none',
                zorder=self.zorder_fill
            )
            self.ax.add_patch(poly)
            
    def draw_borders(self):
        """
        Draw borders for all Venn diagram areas (without fill).
        
        This method uses plot() instead of Polygon for precise control over
        line style and ensures consistent edge rendering across all regions.
        """
        for name, group in self.df_edges.groupby('id'):
            self.ax.plot(
                group['X'],
                group['Y'],
                color=self.border_color,
                linewidth=self.border_linewidth,
                alpha=1.0,
                zorder=self.zorder_border
            )
            
    def add_region_labels(self, fontsize=10, color='black', fontweight='bold'):
        """
        Add labels for each region showing overlap sizes.
        
        Args:
            fontsize (int): Font size for region labels. Default is 10.
            color (str): Color for label text. Default is 'black'.
            fontweight (str): Font weight. Default is 'bold'.
        """
        for _, row in self.df_label.iterrows():
            if pd.notna(row['size']):  # Only label regions with size data
                self.ax.text(
                    row['X'],
                    row['Y'],
                    str(int(row['size'])),
                    fontsize=fontsize,
                    ha='center',
                    va='center',
                    fontweight=fontweight,
                    color=color
                )
                
    def add_set_labels(self, fontsize=12, color='darkred', fontweight='bold'):
        """
        Add labels for each set (e.g., Set A, Set B, etc.).
        
        Args:
            fontsize (int): Font size for set labels. Default is 12.
            color (str): Color for label text. Default is 'darkred'.
            fontweight (str): Font weight. Default is 'bold'.
        """
        for _, row in self.set_label.iterrows():
            self.ax.text(
                row['X'],
                row['Y'],
                str(row['id']),
                fontsize=fontsize,
                ha='center',
                va='center',
                fontweight=fontweight,
                color=color
            )
            
    def plot(self, title="Polygon Based Venn Diagram (Fixed Limits)", add_region_labels=True, add_set_labels=True):
        """
        Complete plotting pipeline for Venn diagram.
        
        This method orchestrates the entire plotting process:
        1. Create figure and axis
        2. Draw filled regions
        3. Draw borders
        4. Add labels
        5. Calculate and set boundaries
        6. Configure styling
        
        Args:
            title (str): Plot title. Default is "Polygon Based Venn Diagram (Fixed Limits)".
            add_region_labels (bool): Whether to add region labels. Default is True.
            add_set_labels (bool): Whether to add set labels. Default is True.
        """
        # Step 1: Create figure and axis
        self.create_figure()
        
        # Step 2: Draw filled regions
        self.draw_filled_regions()
        
        # Step 3: Draw borders
        self.draw_borders()
        
        # Step 4: Add labels
        if add_region_labels:
            self.add_region_labels()
        if add_set_labels:
            self.add_set_labels()
            
        # Step 5: Calculate boundaries and set axis limits
        x_min, x_max, y_min, y_max = self.calculate_boundaries(self.df_edges)
        self.set_axis_limits(x_min, x_max, y_min, y_max)
        
        # Step 6: Configure styling
        self.set_aspect_equal()
        self.hide_axis()
        self.set_title(title)
