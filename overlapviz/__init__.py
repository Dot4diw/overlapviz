"""
OverlapViz - A Python package for drawing Venn diagrams and UpSet plots.

This package provides functionality to visualize set overlaps using:
- Venn diagrams for 2-7 sets
- UpSet plots for 7+ sets

Main functions:
- draw_venn: Create Venn diagrams for 2-7 sets
- draw_upset: Create UpSet plots for 7+ sets
- auto_draw: Automatically choose the appropriate visualization
"""

from .venn import draw_venn
from .upset import draw_upset
from .utils import auto_draw

__version__ = '0.1.0'
__author__ = 'Dot4diw'
__all__ = ['draw_venn', 'draw_upset', 'auto_draw']
