"""
Venn图绘制类

"""

import pickle
from pathlib import Path
from typing import Dict, Optional, Callable

import pandas as pd
import matplotlib.cm as cm
from matplotlib.patches import Polygon
from matplotlib.colors import to_rgba

from base_plot import BasePlot
from plot_config import PlotStyle


class VennPlot(BasePlot):
    """
        venn = VennPlot()
        venn.load_data('geo.pkl', 'overlap.csv')
        venn.plot(title="My Venn")
        venn.show()
    """
    
    def __init__(self, style: Optional[PlotStyle] = None):
        """
        Args:
            style: 绘图样式，默认使用PlotStyle()
        """
        super().__init__(style)
        
        # 数据
        self.df_edges: Optional[pd.DataFrame] = None
        self.df_set_labels: Optional[pd.DataFrame] = None
        self.df_region_labels: Optional[pd.DataFrame] = None
        
        # 自定义选项
        self.custom_colors: Optional[Dict[str, str]] = None
        self.label_formatter: Optional[Callable] = None
    
    def load_geometric_data(self, filepath: str, shape_key: str = 'shape403'):
        """
        Args:
            filepath: pickle文件路径
            shape_key: 数据键名
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        shape_data = data[shape_key]
        self.df_edges = shape_data['set_edge']
        self.df_set_labels = shape_data['set_label']
        self.df_region_labels = shape_data['region_label']
    
    def load_overlap_data(self, filepath: str):
        """
        Args:
            filepath: CSV文件路径
        """
        overlap_data = pd.read_csv(filepath)
        overlap_data['set_names'] = overlap_data['set_names'].str.replace(' & ', '_')
        
        self.df_region_labels = pd.merge(
            self.df_region_labels,
            overlap_data,
            left_on='id',
            right_on='set_names',
            how='left'
        )
    
    def load_data(self, geo_file: str, overlap_file: str, shape_key: str = 'shape404'):
        """
        Args:
            geo_file: 几何数据文件
            overlap_file: 重叠数据文件
            shape_key: 几何数据键名
        """
        self.load_geometric_data(geo_file, shape_key)
        self.load_overlap_data(overlap_file)
    
    def set_custom_colors(self, colors: Dict[str, str]):
        """
        Args:
            colors: {region_id: color} 映射
        """
        self.custom_colors = colors
    
    def set_label_formatter(self, formatter: Callable):
        """
        Args:
            formatter: 格式化函数 func(value) -> str
        """
        self.label_formatter = formatter
    
    def _get_colors(self):
        """生成颜色映射"""
        unique_ids = self.df_edges['id'].unique()
        cmap = cm.get_cmap(self.style.colormap)
        n = len(unique_ids)
        
        colors = {}
        for i, region_id in enumerate(unique_ids):
            if self.custom_colors and region_id in self.custom_colors:
                colors[region_id] = self.custom_colors[region_id]
            else:
                colors[region_id] = cmap(i / max(n - 1, 1))
        
        return colors
    
    def _draw_regions(self):
        """绘制填充区域"""
        colors = self._get_colors()
        
        for region_id, group in self.df_edges.groupby('id'):
            vertices = group[['X', 'Y']].values
            color = colors[region_id]
            
            poly = Polygon(
                vertices,
                closed=True,
                facecolor=to_rgba(color, self.style.fill_alpha),
                edgecolor='none',
                zorder=1
            )
            self.ax.add_patch(poly)
    
    def _draw_borders(self):
        """绘制边框"""
        for region_id, group in self.df_edges.groupby('id'):
            self.ax.plot(
                group['X'],
                group['Y'],
                color=self.style.edge_color,
                linewidth=self.style.edge_width,
                linestyle=self.style.edge_style,
                alpha=self.style.edge_alpha,
                zorder=2
            )
    
    def _draw_region_labels(self):
        """绘制区域标签"""
        if 'size' not in self.df_region_labels.columns:
            return
        
        for _, row in self.df_region_labels.iterrows():
            if pd.notna(row['size']):
                # 格式化文本
                if self.label_formatter:
                    text = self.label_formatter(row['size'])
                else:
                    text = str(int(row['size']))
                
                self.ax.text(
                    row['X'],
                    row['Y'],
                    text,
                    fontsize=self.style.label_fontsize,
                    color=self.style.label_color,
                    fontweight=self.style.label_weight,
                    alpha=self.style.label_alpha,
                    ha=self.style.label_ha,
                    va=self.style.label_va
                )
    
    def _draw_set_labels(self):
        """绘制集合标签"""
        for _, row in self.df_set_labels.iterrows():
            self.ax.text(
                row['X'],
                row['Y'],
                str(row['id']),
                fontsize=self.style.set_label_fontsize,
                color=self.style.set_label_color,
                fontweight=self.style.set_label_weight,
                alpha=self.style.set_label_alpha,
                ha=self.style.set_label_ha,
                va=self.style.set_label_va
            )
    
    def draw(self, 
             title: str = "Venn Diagram",
             show_region_labels: bool = True,
             show_set_labels: bool = True):
        """
        绘制Venn图
        Args:
            title: 标题
            show_region_labels: 是否显示区域标签
            show_set_labels: 是否显示集合标签
        """
        # 创建图形
        self.create_figure()
        
        # 绘制各层
        self._draw_regions()
        self._draw_borders()
        
        if show_region_labels:
            self._draw_region_labels()
        
        if show_set_labels:
            self._draw_set_labels()
        
        # 设置坐标轴
        x_min, x_max, y_min, y_max = self.calculate_limits(self.df_edges)
        self.setup_axes((x_min, x_max), (y_min, y_max))
        
        # 设置标题
        if title:
            self.set_title(title)
    
    def plot(self, title: str = "Venn Diagram", **kwargs):
        """
        绘制并显示
        Args:
            title: 标题
            **kwargs: 传递给draw的其他参数
        """
        self.draw(title=title, **kwargs)
        self.show()
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            'n_regions': len(self.df_edges['id'].unique()),
            'n_sets': len(self.df_set_labels)
        }
        
        if 'size' in self.df_region_labels.columns:
            sizes = self.df_region_labels['size'].dropna()
            stats.update({
                'total_size': sizes.sum(),
                'mean_size': sizes.mean(),
                'max_size': sizes.max(),
                'min_size': sizes.min()
            })
        
        return stats
