"""
绘图配置模块
集中管理所有绘图相关的配置参数
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass
class PlotStyle:
    """绘图样式配置"""
    # 图形设置
    figsize: Tuple[float, float] = (10, 10)
    dpi: int = 100
    facecolor: str = 'white'
    edgecolor: str = 'black'
    
    # 区域样式
    fill_alpha: float = 0.4
    colormap: str = 'viridis'
    
    # 边框样式
    edge_color: str = 'black'
    edge_width: float = 2.0
    edge_style: str = '-'  # 线型: '-' 实线, '--' 虚线, '-.' 点划线, ':' 点线, (0,(5,5)) 自定义
    edge_alpha: float = 1.0
    
    # 标签样式
    label_fontsize: int = 10
    label_color: str = 'black'
    label_weight: str = 'bold'
    label_alpha: float = 1.0
    label_ha: str = 'center'
    label_va: str = 'center'
    
    # 集合标签样式
    set_label_fontsize: int = 12
    set_label_color: str = 'darkred'
    set_label_weight: str = 'bold'
    set_label_alpha: float = 1.0
    set_label_ha: str = 'center'
    set_label_va: str = 'center'
    
    # 标题样式
    title_fontsize: int = 14
    title_weight: str = 'bold'
    title_color: str = 'black'
    title_pad: float = 20.0
    title_loc: str = 'center'
    
    # 坐标轴
    padding_ratio: float = 0.1
    show_axis: bool = False
    axis_color: str = 'black'
    
    # 背景和网格
    show_grid: bool = False
    grid_color: str = 'gray'
    grid_alpha: float = 0.3
    grid_linestyle: str = '--'
    
    @classmethod
    def soft(cls) -> 'PlotStyle':
        """柔和 - 柔和配色，细边框"""
        return cls(
            colormap='Pastel1',
            fill_alpha=0.3,
            edge_color='gray',
            edge_width=1.5,
            edge_alpha=0.8,
            label_fontsize=9,
            label_color='dimgray',
            set_label_color='gray'
        )
    
    @classmethod
    def bold(cls) -> 'PlotStyle':
        """醒目 - 鲜艳配色，粗边框"""
        return cls(
            colormap='Set1',
            fill_alpha=0.6,
            edge_color='black',
            edge_width=3,
            label_fontsize=12,
            label_weight='bold',
            set_label_fontsize=14,
            set_label_weight='bold'
        )
    
    @classmethod
    def paper(cls) -> 'PlotStyle':
        """论文 - 专业配色，高分辨率"""
        return cls(
            dpi=100,
            colormap='tab10',
            fill_alpha=0.4,
            edge_color='black',
            edge_width=1.5,
            label_fontsize=10,
            set_label_fontsize=11,
            title_fontsize=12
        )
    
    @classmethod
    def white(cls) -> 'PlotStyle':
        """白边 - 白色边框，白色标签"""
        return cls(
            colormap='Set2',
            fill_alpha=0.7,
            edge_color='white',
            edge_width=2.5,
            edge_alpha=1.0,
            label_color='white',
            label_weight='bold',
            set_label_color='black',
            set_label_weight='bold',
            title_color='black'
        )
    
    @classmethod
    def clean(cls) -> 'PlotStyle':
        """清爽 - 白色边框，黑色标签"""
        return cls(
            colormap='Set3',
            fill_alpha=0.5,
            edge_color='white',
            edge_width=3.0,
            label_fontsize=11,
            label_color='black',
            set_label_fontsize=13,
            set_label_color='darkblue'
        )
    
    @classmethod
    def dark(cls) -> 'PlotStyle':
        """深色 - 深色背景，白色边框"""
        return cls(
            facecolor='#2e2e2e',
            colormap='Set2',
            fill_alpha=0.6,
            edge_color='white',
            edge_width=2.0,
            label_color='white',
            set_label_color='white',
            title_color='white',
            axis_color='white'
        )
    
    @classmethod
    def pastel(cls) -> 'PlotStyle':
        """粉彩 - 柔和白边框"""
        return cls(
            colormap='Pastel2',
            fill_alpha=0.6,
            edge_color='white',
            edge_width=2.5,
            label_fontsize=10,
            label_color='dimgray',
            set_label_fontsize=12,
            set_label_color='gray'
        )
    
    @classmethod
    def vivid(cls) -> 'PlotStyle':
        """鲜艳 - 高对比度"""
        return cls(
            colormap='bright',
            fill_alpha=0.7,
            edge_color='black',
            edge_width=2.5,
            label_fontsize=11,
            label_weight='bold',
            set_label_fontsize=13,
            set_label_weight='bold'
        )
    
    @classmethod
    def poster(cls) -> 'PlotStyle':
        """海报 - 大字体演示"""
        return cls(
            colormap='Set1',
            fill_alpha=0.6,
            edge_color='black',
            edge_width=3.5,
            label_fontsize=14,
            label_weight='bold',
            set_label_fontsize=16,
            set_label_weight='bold',
            title_fontsize=20,
            title_weight='bold'
        )
    
    @classmethod
    def print(cls) -> 'PlotStyle':
        """打印 - 高DPI打印质量"""
        return cls(
            dpi=100,
            colormap='tab10',
            fill_alpha=0.4,
            edge_color='black',
            edge_width=1.0,
            label_fontsize=9,
            set_label_fontsize=10,
            title_fontsize=11
        )
    
    @classmethod
    def dashed(cls) -> 'PlotStyle':
        """虚线 - 虚线边框"""
        return cls(
            colormap='Set2',
            fill_alpha=0.5,
            edge_color='black',
            edge_width=2.0,
            edge_style='--',
            label_fontsize=10,
            set_label_fontsize=12
        )
    
    @classmethod
    def dotted(cls) -> 'PlotStyle':
        """点线 - 点线边框"""
        return cls(
            colormap='Pastel1',
            fill_alpha=0.4,
            edge_color='dimgray',
            edge_width=2.5,
            edge_style=':',
            label_fontsize=10,
            set_label_fontsize=12
        )
    
    @classmethod
    def dashdot(cls) -> 'PlotStyle':
        """点划线 - 点划线边框"""
        return cls(
            colormap='Set3',
            fill_alpha=0.5,
            edge_color='navy',
            edge_width=2.0,
            edge_style='-.',
            label_fontsize=10,
            set_label_fontsize=12
        )
