"""
快速测试线型主题的可视化效果
"""

from venn_plot import VennPlot
from plot_config import PlotStyle


def test_linestyle_visual():
    """可视化测试线型主题"""
    
    print("生成线型主题示例图...")
    
    linestyle_themes = ['dashed', 'dotted', 'dashdot']
    
    for theme_name in linestyle_themes:
        try:
            print(f"\n生成 {theme_name} 主题...")
            
            style = getattr(PlotStyle, theme_name)()
            venn = VennPlot(style)
            venn.load_data('../geometric_data_v3.pkl', '../plotdata_overlaps.csv', 'shape403')
            
            venn.draw(title=f"{theme_name.upper()} - 线型示例")
            venn.save(f'test_{theme_name}.png')
            venn.close()
            
            print(f"  ✅ 已保存: test_{theme_name}.png")
            print(f"  - 边框颜色: {style.edge_color}")
            print(f"  - 边框线型: {style.edge_style}")
            print(f"  - 边框宽度: {style.edge_width}")
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ 线型主题可视化测试完成！")
    print("请查看生成的图片文件：test_dashed.png, test_dotted.png, test_dashdot.png")


if __name__ == '__main__':
    test_linestyle_visual()
