"""
基类：提供通用的绘图功能，子类继承实现具体绘图逻辑
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from plot_config import PlotStyle


class BasePlot(ABC):
    """
    绘图基类
    
    提供图形创建、坐标轴设置、保存等通用功能
    """
    
    def __init__(self, style: Optional[PlotStyle] = None):
        """
        初始化
        
        Args:
            style: 绘图样式配置，默认使用PlotStyle()
        """
        self.style = style or PlotStyle()
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
    
    def create_figure(self):
        """创建图形和坐标轴"""
        self.fig, self.ax = plt.subplots(
            figsize=self.style.figsize,
            dpi=self.style.dpi,
            facecolor=self.style.facecolor,
            edgecolor=self.style.edgecolor
        )
    
    def calculate_limits(self, df: pd.DataFrame) -> Tuple[float, float, float, float]:
        """
        计算坐标轴范围
        
        Args:
            df: 包含X, Y列的DataFrame
            
        Returns:
            (x_min, x_max, y_min, y_max)
        """
        x_min, x_max = df['X'].min(), df['X'].max()
        y_min, y_max = df['Y'].min(), df['Y'].max()
        
        # 添加边距
        x_margin = (x_max - x_min) * self.style.padding_ratio
        y_margin = (y_max - y_min) * self.style.padding_ratio
        
        return (
            x_min - x_margin,
            x_max + x_margin,
            y_min - y_margin,
            y_max + y_margin
        )
    
    def setup_axes(self, x_lim: Tuple[float, float], y_lim: Tuple[float, float]):
        """
        设置坐标轴
        
        Args:
            x_lim: X轴范围 (min, max)
            y_lim: Y轴范围 (min, max)
        """
        self.ax.set_xlim(x_lim)
        self.ax.set_ylim(y_lim)
        self.ax.set_aspect('equal')
        
        if self.style.show_axis:
            self.ax.spines['top'].set_color(self.style.axis_color)
            self.ax.spines['bottom'].set_color(self.style.axis_color)
            self.ax.spines['left'].set_color(self.style.axis_color)
            self.ax.spines['right'].set_color(self.style.axis_color)
        else:
            self.ax.axis('off')
        
        if self.style.show_grid:
            self.ax.grid(
                True,
                color=self.style.grid_color,
                alpha=self.style.grid_alpha,
                linestyle=self.style.grid_linestyle
            )
    
    def set_title(self, title: str):
        """设置标题"""
        self.ax.set_title(
            title,
            fontsize=self.style.title_fontsize,
            fontweight=self.style.title_weight,
            color=self.style.title_color,
            pad=self.style.title_pad,
            loc=self.style.title_loc
        )
    
    @abstractmethod
    def draw(self, **kwargs):
        """绘制图形（子类实现）"""
        pass
    
    def show(self):
        """显示图形"""
        plt.show()
    
    def save(self, filepath: str, dpi: Optional[int] = None):
        """
        保存图形
        
        Args:
            filepath: 保存路径
            dpi: 分辨率，默认使用style中的dpi
        """
        dpi = dpi or self.style.dpi
        self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    
    def close(self):
        """关闭图形"""
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
