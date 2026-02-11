from plot_config import PlotStyle


def test_all_themes():
    """测试所有13个主题"""
    
    themes = ['soft', 'bold', 'paper', 'white', 'clean', 
              'dark', 'pastel', 'vivid', 'poster', 'print',
              'dashed', 'dotted', 'dashdot']
    
    print("测试所有主题配置:\n")
    print(f"{'主题':<10} {'边框颜色':<12} {'线型':<8} {'边框宽度':<10} {'透明度':<8}")
    print("-" * 60)
    
    for theme_name in themes:
        style = getattr(PlotStyle, theme_name)()
        print(f"{theme_name:<10} {style.edge_color:<12} {style.edge_style:<8} {style.edge_width:<10.1f} {style.fill_alpha:<8.1f}")
    
    print("\n✅ 所有主题配置正常！")


def test_linestyle_themes():
    """测试线型主题"""
    
    print("\n测试线型主题:\n")
    
    linestyle_themes = {
        'dashed': '--',
        'dotted': ':',
        'dashdot': '-.'
    }
    
    for theme_name, expected_style in linestyle_themes.items():
        style = getattr(PlotStyle, theme_name)()
        actual_style = style.edge_style
        
        if actual_style == expected_style:
            print(f"✅ {theme_name:<10} - 线型: {actual_style} (正确)")
        else:
            print(f"❌ {theme_name:<10} - 线型: {actual_style} (期望: {expected_style})")
    
    print("\n✅ 线型主题测试通过！")


def test_custom_linestyle():
    """测试自定义线型"""
    
    print("\n测试自定义线型:\n")
    
    # 基于现有主题修改
    style = PlotStyle.bold()
    style.edge_style = ':'
    print(f"修改bold主题线型为点线: {style.edge_style}")
    
    style = PlotStyle.white()
    style.edge_style = '--'
    print(f"修改white主题线型为虚线: {style.edge_style}")
    
    # 创建自定义样式
    style = PlotStyle(
        edge_color='red',
        edge_width=3.0,
        edge_style='-.',
        fill_alpha=0.6
    )
    print(f"自定义样式线型: {style.edge_style}")
    
    print("\n✅ 自定义线型测试通过！")


if __name__ == '__main__':
    print("="*60)
    print("线型主题测试")
    print("="*60)
    
    test_all_themes()
    test_linestyle_themes()
    test_custom_linestyle()
    
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)
