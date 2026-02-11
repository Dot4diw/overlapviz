"""
使用示例
"""

from venn_plot import VennPlot
from plot_config import PlotStyle


def example_1_basic():
    """示例1: 基础使用"""
    print("示例1: 基础使用")
    
    venn = VennPlot()
    venn.load_data('../geometric_data_v3.pkl', '../plotdata_overlaps.csv', 'shape403')
    venn.plot(title="基础Venn图")


def example_2_custom_style():
    """示例2: 自定义样式"""
    print("示例2: 自定义样式")
    
    # 创建自定义样式
    style = PlotStyle(
        figsize=(12, 10),
        colormap='Set3',
        fill_alpha=0.5,
        edge_color='navy',
        edge_width=2.5,
        label_fontsize=11
    )
    
    venn = VennPlot(style)
    venn.load_data('../geometric_data_v3.pkl', '../plotdata_overlaps.csv', 'shape403')
    venn.plot(title="自定义样式")


def example_3_preset_styles():
    """示例3: 预设样式"""
    print("示例3: 预设样式")
    
    styles = {
        'soft': PlotStyle.soft(),
        'bold': PlotStyle.bold(),
        'paper': PlotStyle.paper(),
        'dashed': PlotStyle.dashed(),
        'dotted': PlotStyle.dotted()
    }
    
    for name, style in styles.items():
        venn = VennPlot(style)
        venn.load_data('../geometric_data_v3.pkl', '../plotdata_overlaps.csv', 'shape403')
        venn.draw(title=f"{name.title()} 风格")
        venn.show()
        venn.close()


def example_4_custom_colors():
    """示例4: 自定义颜色"""
    print("示例4: 自定义颜色")
    
    venn = VennPlot()
    venn.load_data('../geometric_data_v3.pkl', '../plotdata_overlaps.csv', 'shape403')
    
    # 设置自定义颜色（根据实际region_id调整）
    # custom_colors = {
    #     'region1': '#FF6B6B',
    #     'region2': '#4ECDC4'
    # }
    # venn.set_custom_colors(custom_colors)
    
    venn.plot(title="自定义颜色")


def example_5_custom_formatter():
    """示例5: 自定义标签格式"""
    print("示例5: 自定义标签格式")
    
    def format_with_comma(value):
        """添加千位分隔符"""
        return f"{int(value):,}"
    
    venn = VennPlot()
    venn.load_data('../geometric_data_v3.pkl', '../plotdata_overlaps.csv', 'shape403')
    venn.set_label_formatter(format_with_comma)
    venn.plot(title="自定义标签格式")


def example_6_save():
    """示例6: 保存图形"""
    print("示例6: 保存图形")
    
    style = PlotStyle.paper()
    venn = VennPlot(style)
    venn.load_data('../geometric_data_v3.pkl', '../plotdata_overlaps.csv', 'shape403')
    
    venn.draw(title="保存示例")
    venn.save('output.png', dpi=300)
    venn.show()
    venn.close()
    
    print("已保存到 output.png")


def example_7_statistics():
    """示例7: 统计信息"""
    print("示例7: 统计信息")
    
    venn = VennPlot()
    venn.load_data('../geometric_data_v3.pkl', '../plotdata_overlaps.csv', 'shape403')
    
    stats = venn.get_statistics()
    print("\n统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    venn.plot(title="带统计信息")


def example_8_batch():
    """示例8: 批量生成"""
    print("示例8: 批量生成")
    
    shape_keys = ['shape403', 'shape404']
    style = PlotStyle.paper()
    
    for key in shape_keys:
        try:
            venn = VennPlot(style)
            venn.load_data('../geometric_data_v3.pkl', '../plotdata_overlaps.csv', key)
            venn.draw(title=f"Venn - {key}")
            venn.save(f'output_{key}.png')
            venn.close()
            print(f"已保存: output_{key}.png")
        except Exception as e:
            print(f"处理 {key} 时出错: {e}")


def example_9_linestyles():
    """示例9: 不同线型"""
    print("示例9: 不同线型")
    
    # 使用预设线型主题
    linestyles = {
        'dashed': PlotStyle.dashed(),
        'dotted': PlotStyle.dotted(),
        'dashdot': PlotStyle.dashdot()
    }
    
    for name, style in linestyles.items():
        venn = VennPlot(style)
        venn.load_data('../geometric_data_v3.pkl', '../plotdata_overlaps.csv', 'shape403')
        venn.draw(title=f"{name.title()} 线型")
        venn.show()
        venn.close()


def example_10_custom_linestyle():
    """示例10: 自定义线型"""
    print("示例10: 自定义线型")
    
    # 基于现有主题修改线型
    style = PlotStyle.bold()
    style.edge_style = ':'  # 改为点线
    style.edge_width = 3.0
    style.edge_color = 'red'
    
    venn = VennPlot(style)
    venn.load_data('../geometric_data_v3.pkl', '../plotdata_overlaps.csv', 'shape403')
    venn.plot(title="自定义点线边框")


if __name__ == '__main__':
    print("Venn图绘制系统 - 简化版示例\n")
    
    # 运行示例1
    example_1_basic()
    
    # 取消注释运行其他示例
    # example_2_custom_style()
    # example_3_preset_styles()
    # example_4_custom_colors()
    # example_5_custom_formatter()
    # example_6_save()
    # example_7_statistics()
    # example_8_batch()
    # example_9_linestyles()
    # example_10_custom_linestyle()
