from venn_plot import VennPlot
from plot_config import PlotStyle


def showcase_all_themes():
    """展示所有预设主题"""
    
    themes = {
        'soft': '柔和 - 柔和配色，细边框',
        'bold': '醒目 - 鲜艳配色，粗边框',
        'paper': '论文 - 专业配色，高分辨率',
        'white': '白边 - 白色边框，白色标签',
        'clean': '清爽 - 白色边框，黑色标签',
        'dark': '深色 - 深色背景，白色边框',
        'pastel': '粉彩 - 柔和白边框',
        'vivid': '鲜艳 - 高对比度',
        'poster': '海报 - 大字体演示',
        'print': '打印 - 高DPI打印质量',
        'dashed': '虚线 - 虚线边框',
        'dotted': '点线 - 点线边框',
        'dashdot': '点划线 - 点划线边框'
    }
    
    print("="*60)
    print("Venn图主题展示")
    print("="*60)
    print(f"\n共有 {len(themes)} 个预设主题:\n")
    
    for i, (theme_name, desc) in enumerate(themes.items(), 1):
        print(f"{i:2}. {theme_name:8} - {desc}")
    
    print("\n" + "="*60)
    print("开始展示各主题...")
    print("="*60)
    
    for theme_name, desc in themes.items():
        print(f"\n展示主题: {theme_name}")
        print(f"描述: {desc}")
        
        try:
            # 获取主题
            style = getattr(PlotStyle, theme_name)()
            
            # 创建Venn图
            venn = VennPlot(style)
            venn.load_data('../geometric_data_v3.pkl', '../plotdata_overlaps.csv', 'shape403')
            
            # 绘制
            venn.draw(title=f"{theme_name.upper()} - {desc.split(' - ')[0]}")
            
            # 显示配置信息
            print(f"  配置:")
            print(f"    - 图形大小: {style.figsize}")
            print(f"    - 颜色方案: {style.colormap}")
            print(f"    - 边框颜色: {style.edge_color}")
            print(f"    - 边框宽度: {style.edge_width}")
            print(f"    - 边框线型: {style.edge_style}")
            print(f"    - 透明度: {style.fill_alpha}")
            print(f"    - DPI: {style.dpi}")
            
            # 保存
            output_file = f'theme_{theme_name}.png'
            venn.save(output_file)
            print(f"  ✅ 已保存: {output_file}")
            
            # 显示
            venn.show()
            venn.close()
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("主题展示完成！")
    print("="*60)


def showcase_white_themes():
    """展示白色边框主题"""
    
    print("\n" + "="*60)
    print("白色边框主题对比")
    print("="*60)
    
    white_themes = {
        'white': '白边 - 白色边框，白色标签',
        'clean': '清爽 - 白色边框，黑色标签',
        'dark': '深色 - 深色背景，白色边框',
        'pastel': '粉彩 - 柔和白边框'
    }
    
    for theme_name, desc in white_themes.items():
        print(f"\n{desc} ({theme_name}):")
        style = getattr(PlotStyle, theme_name)()
        
        print(f"  图形大小: {style.figsize}")
        print(f"  边框颜色: {style.edge_color}")
        print(f"  边框宽度: {style.edge_width}")
        print(f"  标签颜色: {style.label_color}")
        print(f"  集合标签颜色: {style.set_label_color}")
        print(f"  背景色: {style.facecolor}")
        
        venn = VennPlot(style)
        venn.load_data('../geometric_data_v3.pkl', '../plotdata_overlaps.csv', 'shape403')
        venn.draw(title=f"{theme_name.upper()} - {desc.split(' - ')[0]}")
        venn.save(f'white_{theme_name}.png')
        venn.show()
        venn.close()
        
        print(f"  ✅ 已保存: white_{theme_name}.png")


def showcase_by_category():
    """按类别展示主题"""
    
    print("\n" + "="*60)
    print("按类别展示主题")
    print("="*60)
    
    categories = {
        '日常使用': ['soft', 'bold'],
        '学术论文': ['paper', 'print'],
        '白色边框': ['white', 'clean', 'dark', 'pastel'],
        '演示展示': ['vivid', 'poster'],
        '线型效果': ['dashed', 'dotted', 'dashdot']
    }
    
    for category, theme_list in categories.items():
        print(f"\n【{category}】")
        for theme_name in theme_list:
            style = getattr(PlotStyle, theme_name)()
            print(f"  {theme_name:8} - 边框: {style.edge_color:8} 线型: {style.edge_style:4} 透明度: {style.fill_alpha} DPI: {style.dpi}")


def compare_themes():
    """对比不同主题"""
    
    print("\n" + "="*60)
    print("主题详细对比")
    print("="*60)
    
    themes = ['soft', 'bold', 'paper', 'white', 'clean', 
              'dark', 'pastel', 'vivid', 'poster', 'print',
              'dashed', 'dotted', 'dashdot']
    
    print(f"\n{'主题':<10} {'边框颜色':<10} {'线型':<8} {'边框宽度':<10} {'透明度':<10} {'DPI':<10}")
    print("-" * 70)
    
    for theme_name in themes:
        style = getattr(PlotStyle, theme_name)()
        print(f"{theme_name:<10} {style.edge_color:<10} {style.edge_style:<8} {style.edge_width:<10.1f} {style.fill_alpha:<10.1f} {style.dpi:<10}")


def quick_demo():
    """快速演示 - 只展示几个代表性主题"""
    
    print("\n" + "="*60)
    print("快速演示 - 代表性主题")
    print("="*60)
    
    demo_themes = {
        'soft': '柔和',
        'white': '白边',
        'paper': '论文'
    }
    
    for theme_name, desc in demo_themes.items():
        print(f"\n展示: {theme_name} ({desc})")
        
        style = getattr(PlotStyle, theme_name)()
        venn = VennPlot(style)
        venn.load_data('../geometric_data_v3.pkl', '../plotdata_overlaps.csv', 'shape403')
        venn.draw(title=f"{theme_name.upper()} - {desc}")
        venn.save(f'demo_{theme_name}.png')
        venn.show()
        venn.close()
        
        print(f"  ✅ 已保存: demo_{theme_name}.png")


def showcase_linestyles():
    """展示线型主题"""
    
    print("\n" + "="*60)
    print("线型主题展示")
    print("="*60)
    
    linestyle_themes = {
        'dashed': '虚线 - 虚线边框',
        'dotted': '点线 - 点线边框',
        'dashdot': '点划线 - 点划线边框'
    }
    
    for theme_name, desc in linestyle_themes.items():
        print(f"\n{desc} ({theme_name}):")
        style = getattr(PlotStyle, theme_name)()
        
        print(f"  边框颜色: {style.edge_color}")
        print(f"  边框宽度: {style.edge_width}")
        print(f"  边框线型: {style.edge_style}")
        print(f"  透明度: {style.fill_alpha}")
        
        venn = VennPlot(style)
        venn.load_data('../geometric_data_v3.pkl', '../plotdata_overlaps.csv', 'shape403')
        venn.draw(title=f"{theme_name.upper()} - {desc.split(' - ')[0]}")
        venn.save(f'linestyle_{theme_name}.png')
        venn.show()
        venn.close()
        
        print(f"  ✅ 已保存: linestyle_{theme_name}.png")


def batch_generate():
    """批量生成所有主题的图片"""
    
    print("\n" + "="*60)
    print("批量生成所有主题")
    print("="*60)
    
    themes = ['soft', 'bold', 'paper', 'white', 'clean', 
              'dark', 'pastel', 'vivid', 'poster', 'print',
              'dashed', 'dotted', 'dashdot']
    
    for theme_name in themes:
        try:
            style = getattr(PlotStyle, theme_name)()
            venn = VennPlot(style)
            venn.load_data('../geometric_data_v3.pkl', '../plotdata_overlaps.csv', 'shape403')
            venn.draw(title=f"{theme_name.upper()}")
            venn.save(f'batch_{theme_name}.png')
            venn.close()
            print(f"  ✅ {theme_name:8} - 已保存: batch_{theme_name}.png")
        except Exception as e:
            print(f"  ❌ {theme_name:8} - 错误: {e}")
    
    print("\n批量生成完成！")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("Venn图主题展示系统")
    print("="*60)
    print("\n选择展示模式:")
    print("1. 展示所有主题（逐个显示）")
    print("2. 展示白色边框主题")
    print("3. 按类别展示")
    print("4. 对比所有主题")
    print("5. 快速演示（3个代表性主题）")
    print("6. 批量生成（不显示，只保存）")
    print("7. 展示线型主题")
    print("8. 全部运行")
    
    choice = input("\n请选择 (1-8，默认5): ").strip() or "5"
    
    if choice == "1":
        showcase_all_themes()
    elif choice == "2":
        showcase_white_themes()
    elif choice == "3":
        showcase_by_category()
    elif choice == "4":
        compare_themes()
    elif choice == "5":
        quick_demo()
    elif choice == "6":
        batch_generate()
    elif choice == "7":
        showcase_linestyles()
    elif choice == "8":
        showcase_by_category()
        compare_themes()
        showcase_white_themes()
        showcase_linestyles()
        batch_generate()
    else:
        print("无效选择，运行快速演示")
        quick_demo()


if __name__ == '__main__':
    main()
