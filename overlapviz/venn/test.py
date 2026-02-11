"""
测试脚本
"""

import sys
from pathlib import Path

from venn_plot import VennPlot
from plot_config import PlotStyle


def test_basic():
    """测试基础功能"""
    print("测试1: 基础功能")
    try:
        venn = VennPlot()
        venn.load_data('../geometric_data_v3.pkl', '../plotdata_overlaps.csv', 'shape403')
        
        assert venn.df_edges is not None
        assert venn.df_set_labels is not None
        assert venn.df_region_labels is not None
        
        print("  ✅ 数据加载成功")
        
        stats = venn.get_statistics()
        assert 'n_regions' in stats
        assert 'n_sets' in stats
        print(f"  ✅ 统计信息: {stats}")
        
        venn.draw(title="测试")
        print("  ✅ 绘图成功")
        
        venn.close()
        print("  ✅ 资源清理成功")
        
        return True
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        return False


def test_styles():
    """测试预设样式"""
    print("\n测试2: 预设样式")
    try:
        styles = {
            'minimal': PlotStyle.minimal(),
            'bold': PlotStyle.bold(),
            'scientific': PlotStyle.scientific()
        }
        
        for name, style in styles.items():
            venn = VennPlot(style)
            venn.load_data('../geometric_data_v3.pkl', '../plotdata_overlaps.csv', 'shape403')
            venn.draw(title=f"{name} 测试")
            venn.close()
            print(f"  ✅ {name} 样式成功")
        
        return True
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        return False


def test_custom():
    """测试自定义功能"""
    print("\n测试3: 自定义功能")
    try:
        # 自定义样式
        style = PlotStyle(
            figsize=(12, 10),
            colormap='Set3',
            fill_alpha=0.5
        )
        venn = VennPlot(style)
        venn.load_data('../geometric_data_v3.pkl', '../plotdata_overlaps.csv', 'shape403')
        print("  ✅ 自定义样式成功")
        
        # 自定义格式化
        venn.set_label_formatter(lambda x: f"{int(x):,}")
        print("  ✅ 自定义格式化成功")
        
        venn.draw(title="自定义测试")
        venn.close()
        
        return True
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        return False


def test_save():
    """测试保存功能"""
    print("\n测试4: 保存功能")
    try:
        venn = VennPlot()
        venn.load_data('../geometric_data_v3.pkl', '../plotdata_overlaps.csv', 'shape403')
        venn.draw(title="保存测试")
        
        output_file = 'test_output.png'
        venn.save(output_file)
        
        if Path(output_file).exists():
            print(f"  ✅ 文件保存成功: {output_file}")
            Path(output_file).unlink()
            print("  ✅ 测试文件已清理")
        else:
            print(f"  ❌ 文件保存失败")
            return False
        
        venn.close()
        return True
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("="*50)
    print("Venn图绘制系统 - 简化版测试")
    print("="*50)
    
    tests = [
        test_basic,
        test_styles,
        test_custom,
        test_save
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "="*50)
    print("测试结果")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return 1


if __name__ == '__main__':
    sys.exit(main())
