"""
Refactored Venn Diagram Plotting Example.

This script demonstrates how to use the VennPlot class to create
polygon-based Venn diagrams with overlap data integration.
"""

from venn_plot import VennPlot


def main():
    """
    Main function to create and display a Venn diagram.
    
    This example demonstrates the complete workflow:
    1. Initialize VennPlot with custom styling
    2. Load geometric data from pickle file
    3. Load overlap data from CSV file
    4. Create the complete plot
    5. Display the result
    """
    # Initialize VennPlot with custom styling parameters
    venn = VennPlot(
        figsize=(10, 10),
        zorder_fill=1,          # Place fills in background
        zorder_border=2,       # Place borders on top of fills
        border_linewidth=2,     # Border thickness
        border_color='black',   # Border color
        fill_alpha=0.4,         # Fill transparency
        colormap='viridis'      # Color scheme
    )
    
    # Load geometric data from pickle file
    venn.load_geometric_data('geometric_data_v3.pkl', shape_key='shape403')
    
    # Load overlap data and merge with region labels
    venn.load_overlap_data('plotdata_overlaps.csv')
    
    # Print loaded data for verification
    print("Overlap Data:")
    print(venn.overlap_data)
    print("\nMerged Region Labels:")
    print(venn.df_label)
    
    # Create the complete Venn diagram
    venn.plot(
        title="Polygon Based Venn Diagram (Fixed Limits)",
        add_region_labels=True,
        add_set_labels=True
    )
    
    # Display the plot
    venn.show()
    
    # Optional: Save to file
    # venn.save('venn_diagram.png', dpi=300)
    
    # Optional: Clean up memory
    # venn.close()


if __name__ == '__main__':
    main()
