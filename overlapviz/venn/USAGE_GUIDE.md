# 使用指南

## 快速索引

- [基础使用](#基础使用)
- [13个主题](#13个主题)
- [白色边框](#白色边框)
- [线型主题](#线型主题)
- [自定义样式](#自定义样式)
- [高级功能](#高级功能)
- [场景推荐](#场景推荐)

## 基础使用

```python
from venn_plot import VennPlot

venn = VennPlot()
venn.load_data('geo.pkl', 'overlap.csv')
venn.plot(title="我的Venn图")
```

## 13个主题

### 一行代码切换主题

```python
from venn_plot import VennPlot
from plot_config import PlotStyle

# 柔和 - 日常查看
venn = VennPlot(PlotStyle.soft())

# 醒目 - 演示展示
venn = VennPlot(PlotStyle.bold())

# 论文 - 学术论文
venn = VennPlot(PlotStyle.paper())

# 白边 - 深色背景
venn = VennPlot(PlotStyle.white())

# 清爽 - 清新风格
venn = VennPlot(PlotStyle.clean())

# 深色 - 深色模式
venn = VennPlot(PlotStyle.dark())

# 粉彩 - 温和风格
venn = VennPlot(PlotStyle.pastel())

# 鲜艳 - 醒目展示
venn = VennPlot(PlotStyle.vivid())

# 海报 - PPT演示
venn = VennPlot(PlotStyle.poster())

# 打印 - 高质量打印
venn = VennPlot(PlotStyle.print())

# 虚线 - 虚线边框
venn = VennPlot(PlotStyle.dashed())

# 点线 - 点线边框
venn = VennPlot(PlotStyle.dotted())

# 点划线 - 点划线边框
venn = VennPlot(PlotStyle.dashdot())

# 鲜艳 - 醒目展示
venn = VennPlot(PlotStyle.vivid())

# 海报 - PPT演示
venn = VennPlot(PlotStyle.poster())

# 打印 - 高质量打印
venn = VennPlot(PlotStyle.print())
```

### 主题对比

| 主题 | 边框颜色 | 边框宽度 | 透明度 | DPI | 适用场景 |
|------|----------|----------|--------|-----|----------|
| soft | 灰色 | 1.5 | 0.3 | 100 | 日常查看 |
| bold | 黑色 | 3.0 | 0.6 | 100 | 演示展示 |
| paper | 黑色 | 1.5 | 0.4 | 100 | 学术论文 |
| white | 白色 | 2.5 | 0.7 | 100 | 深色背景 |
| clean | 白色 | 3.0 | 0.5 | 100 | 清新风格 |
| dark | 白色 | 2.0 | 0.6 | 100 | 深色模式 |
| pastel | 白色 | 2.5 | 0.6 | 100 | 温和风格 |
| vivid | 黑色 | 2.5 | 0.7 | 100 | 醒目展示 |
| poster | 黑色 | 3.5 | 0.6 | 100 | PPT演示 |
| print | 黑色 | 1.0 | 0.4 | 100 | 高质量打印 |

## 白色边框

4个白色边框主题，适合不同场景：

```python
# white - 白边框+白标签，适合深色背景
venn = VennPlot(PlotStyle.white())

# clean - 白边框+黑标签，清爽风格
venn = VennPlot(PlotStyle.clean())

# dark - 白边框+深色背景，深色模式
venn = VennPlot(PlotStyle.dark())

# pastel - 白边框+柔和配色，温和风格
venn = VennPlot(PlotStyle.pastel())
```

## 线型主题

3个线型主题，提供不同的边框样式：

```python
# dashed - 虚线边框，适合区分效果
venn = VennPlot(PlotStyle.dashed())

# dotted - 点线边框，柔和区分
venn = VennPlot(PlotStyle.dotted())

# dashdot - 点划线边框，特殊标记
venn = VennPlot(PlotStyle.dashdot())
```

### 可用线型

```python
# 实线（默认）
edge_style = '-'

# 虚线
edge_style = '--'

# 点线
edge_style = ':'

# 点划线
edge_style = '-.'

# 自定义模式：(偏移, (线长, 间隔))
edge_style = (0, (5, 5))        # 5像素线，5像素间隔
edge_style = (0, (3, 1, 1, 1))  # 复杂模式
```

### 自定义线型示例

```python
# 基于现有主题修改线型
style = PlotStyle.bold()
style.edge_style = ':'  # 改为点线
style.edge_width = 3.0

venn = VennPlot(style)
venn.load_data('geo.pkl', 'overlap.csv')
venn.plot()

# 创建自定义线型
style = PlotStyle(
    edge_color='red',
    edge_width=2.5,
    edge_style='--',  # 虚线
    fill_alpha=0.6
)

venn = VennPlot(style)
venn.load_data('geo.pkl', 'overlap.csv')
venn.plot()
```

## 自定义样式

### 方式1: 完全自定义

```python
from plot_config import PlotStyle

style = PlotStyle(
    figsize=(12, 10),
    colormap='Set3',
    fill_alpha=0.5,
    edge_color='navy',
    edge_width=2.5,
    label_fontsize=11
)

venn = VennPlot(style)
```

### 方式2: 基于主题修改

```python
# 基于white主题修改
style = PlotStyle.white()
style.edge_width = 4.0      # 加粗边框
style.fill_alpha = 0.8      # 增加透明度
style.label_fontsize = 12   # 增大字体

venn = VennPlot(style)
```

## 高级功能

### 自定义颜色

```python
venn = VennPlot()
venn.load_data('geo.pkl', 'overlap.csv')

# 设置自定义颜色
custom_colors = {
    'SetA': '#FF6B6B',
    'SetB': '#4ECDC4',
    'SetA_SetB': '#95E1D3'
}
venn.set_custom_colors(custom_colors)

venn.plot()
```

### 自定义标签格式

```python
# 添加千位分隔符
def format_with_comma(value):
    return f"{int(value):,}"

venn = VennPlot()
venn.load_data('geo.pkl', 'overlap.csv')
venn.set_label_formatter(format_with_comma)
venn.plot()

# 显示百分比
def format_with_percent(value):
    total = 1000
    percent = (value / total) * 100
    return f"{int(value)}\n({percent:.1f}%)"

venn.set_label_formatter(format_with_percent)
```

### 保存图形

```python
venn = VennPlot()
venn.load_data('geo.pkl', 'overlap.csv')

venn.draw(title="My Venn")

# 保存为PNG
venn.save('output.png', dpi=300)

# 保存为PDF
venn.save('output.pdf')

# 保存为SVG
venn.save('output.svg')

venn.show()
venn.close()
```

### 获取统计信息

```python
venn = VennPlot()
venn.load_data('geo.pkl', 'overlap.csv')

# 获取统计数据
stats = venn.get_statistics()
print(f"区域数: {stats['n_regions']}")
print(f"集合数: {stats['n_sets']}")
print(f"总大小: {stats['total_size']}")
print(f"平均大小: {stats['mean_size']:.2f}")
```

### 批量生成

```python
# 使用同一主题生成多个图
style = PlotStyle.paper()

for key in ['shape403', 'shape404', 'shape405']:
    venn = VennPlot(style)
    venn.load_data('geo.pkl', 'overlap.csv', key)
    venn.draw(title=f"Venn - {key}")
    venn.save(f'output_{key}.png')
    venn.close()
```

## 场景推荐

### 日常查看
```python
PlotStyle.soft()    # 柔和舒适
```

### 演示展示
```python
PlotStyle.bold()    # 醒目
PlotStyle.poster()  # 大字体
```

### 学术论文
```python
PlotStyle.paper()   # 专业，300 DPI
PlotStyle.print()   # 高质量，600 DPI
```

### 深色背景
```python
PlotStyle.white()   # 白边框+白标签
PlotStyle.dark()    # 深色背景
```

### 清新风格
```python
PlotStyle.clean()   # 清爽
PlotStyle.pastel()  # 粉彩
```

### 网页展示
```python
PlotStyle.clean()   # 清爽
PlotStyle.soft()    # 柔和
```

## 完整示例

### 示例1: 基础使用

```python
from venn_plot import VennPlot

venn = VennPlot()
venn.load_data('geo.pkl', 'overlap.csv')
venn.plot(title="基础Venn图")
```

### 示例2: 使用白边主题

```python
from venn_plot import VennPlot
from plot_config import PlotStyle

venn = VennPlot(PlotStyle.white())
venn.load_data('geo.pkl', 'overlap.csv')
venn.plot(title="白边Venn图")
```

### 示例3: 自定义并保存

```python
from venn_plot import VennPlot
from plot_config import PlotStyle

style = PlotStyle.paper()
style.edge_width = 2.0

venn = VennPlot(style)
venn.load_data('geo.pkl', 'overlap.csv')
venn.draw(title="论文Venn图")
venn.save('paper.png', dpi=300)
venn.show()
venn.close()
```

### 示例4: 批量生成不同主题

```python
from venn_plot import VennPlot
from plot_config import PlotStyle

themes = ['soft', 'bold', 'white', 'clean']

for theme_name in themes:
    style = getattr(PlotStyle, theme_name)()
    venn = VennPlot(style)
    venn.load_data('geo.pkl', 'overlap.csv')
    venn.draw(title=f"{theme_name.upper()}")
    venn.save(f'{theme_name}.png')
    venn.close()
```

## 运行示例

```bash
# 查看所有示例
python example.py

# 展示所有主题
python showcase.py

# 测试主题
python test_themes.py
```

## 获取帮助

- `THEMES.md` - 完整主题参考
- `QUICK_START.md` - 快速开始
- `README_NEW.md` - 完整文档
- `showcase.py` - 主题展示
