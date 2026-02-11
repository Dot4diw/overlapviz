# 线型使用指南

## 概述

Venn图绘制系统支持多种边框线型，包括实线、虚线、点线、点划线等。

## 3个预设线型主题

### dashed - 虚线
```python
from venn_plot import VennPlot
from plot_config import PlotStyle

venn = VennPlot(PlotStyle.dashed())
venn.load_data('geo.pkl', 'overlap.csv')
venn.plot(title="虚线边框")
```

特点：
- 黑色虚线边框（`--`）
- Set2配色
- 边框宽度：2.0
- 适合区分不同区域

### dotted - 点线
```python
venn = VennPlot(PlotStyle.dotted())
venn.load_data('geo.pkl', 'overlap.csv')
venn.plot(title="点线边框")
```

特点：
- 灰色点线边框（`:`）
- Pastel1配色
- 边框宽度：2.5
- 柔和的区分效果

### dashdot - 点划线
```python
venn = VennPlot(PlotStyle.dashdot())
venn.load_data('geo.pkl', 'overlap.csv')
venn.plot(title="点划线边框")
```

特点：
- 深蓝点划线边框（`-.`）
- Set3配色
- 边框宽度：2.0
- 特殊标记效果

## 可用线型

### 基础线型

| 线型代码 | 说明 | 示例 |
|---------|------|------|
| `-` | 实线 | ━━━━━━ |
| `--` | 虚线 | ╍╍╍╍╍╍ |
| `:` | 点线 | ┄┄┄┄┄┄ |
| `-.` | 点划线 | ╍━╍━╍━ |

### 自定义线型

使用元组格式：`(偏移, (线长, 间隔, ...))`

```python
# 5像素线，5像素间隔
edge_style = (0, (5, 5))

# 10像素线，2像素间隔
edge_style = (0, (10, 2))

# 复杂模式：3线-1空-1线-1空
edge_style = (0, (3, 1, 1, 1))

# 长短虚线：10线-2空-2线-2空
edge_style = (0, (10, 2, 2, 2))
```

## 使用方法

### 方法1: 使用预设主题

```python
from venn_plot import VennPlot
from plot_config import PlotStyle

# 直接使用预设线型主题
venn = VennPlot(PlotStyle.dashed())
venn.load_data('geo.pkl', 'overlap.csv')
venn.plot()
```

### 方法2: 修改现有主题

```python
# 基于bold主题，改为虚线
style = PlotStyle.bold()
style.edge_style = '--'
style.edge_width = 3.0

venn = VennPlot(style)
venn.load_data('geo.pkl', 'overlap.csv')
venn.plot()
```

### 方法3: 完全自定义

```python
# 创建自定义样式
style = PlotStyle(
    colormap='Set3',
    fill_alpha=0.5,
    edge_color='red',
    edge_width=2.5,
    edge_style=':',  # 点线
    label_fontsize=11
)

venn = VennPlot(style)
venn.load_data('geo.pkl', 'overlap.csv')
venn.plot()
```

## 线型组合示例

### 示例1: 虚线+白色边框

```python
style = PlotStyle.white()
style.edge_style = '--'
style.edge_width = 3.0

venn = VennPlot(style)
venn.load_data('geo.pkl', 'overlap.csv')
venn.plot(title="白色虚线边框")
```

### 示例2: 点线+粗边框

```python
style = PlotStyle.bold()
style.edge_style = ':'
style.edge_width = 4.0
style.edge_color = 'navy'

venn = VennPlot(style)
venn.load_data('geo.pkl', 'overlap.csv')
venn.plot(title="深蓝粗点线")
```

### 示例3: 自定义虚线模式

```python
style = PlotStyle(
    edge_color='darkgreen',
    edge_width=2.0,
    edge_style=(0, (10, 3)),  # 10像素线，3像素间隔
    fill_alpha=0.6
)

venn = VennPlot(style)
venn.load_data('geo.pkl', 'overlap.csv')
venn.plot(title="自定义虚线模式")
```

## 应用场景

### 区分不同数据集

使用不同线型区分多个Venn图：

```python
datasets = {
    'dataset1': PlotStyle.dashed(),   # 虚线
    'dataset2': PlotStyle.dotted(),   # 点线
    'dataset3': PlotStyle.dashdot()   # 点划线
}

for name, style in datasets.items():
    venn = VennPlot(style)
    venn.load_data('geo.pkl', f'{name}.csv')
    venn.draw(title=f"{name}")
    venn.save(f'{name}.png')
    venn.close()
```

### 强调特定区域

```python
# 主图用实线
style_main = PlotStyle.bold()

# 对比图用虚线
style_compare = PlotStyle.bold()
style_compare.edge_style = '--'
```

### 打印友好

```python
# 黑白打印时，使用不同线型区分
style = PlotStyle.print()
style.edge_style = '--'  # 虚线更容易在黑白打印中区分
```

## 线型对比

| 主题 | 边框颜色 | 线型 | 边框宽度 | 适用场景 |
|------|----------|------|----------|----------|
| soft | 灰色 | 实线 | 1.5 | 日常查看 |
| bold | 黑色 | 实线 | 3.0 | 演示展示 |
| dashed | 黑色 | 虚线 | 2.0 | 区分效果 |
| dotted | 灰色 | 点线 | 2.5 | 柔和区分 |
| dashdot | 深蓝 | 点划线 | 2.0 | 特殊标记 |

## 注意事项

1. 线型在小尺寸图片中可能不明显，建议使用较粗的边框（edge_width >= 2.0）
2. 点线（`:`）在某些情况下可能显示为很小的点，可以增加边框宽度
3. 自定义线型模式的数值单位是像素
4. 线型与边框颜色、宽度配合使用效果更好
5. 在深色背景下，建议使用白色或浅色边框配合线型

## 测试

运行测试脚本查看所有线型效果：

```bash
cd refactored
python test_linestyles.py        # 配置测试
python quick_test_linestyles.py  # 可视化测试
```

## 更多示例

查看完整示例：

```bash
cd refactored
python example.py  # 取消注释 example_9_linestyles() 和 example_10_custom_linestyle()
```
