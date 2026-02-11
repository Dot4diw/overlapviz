# Venn图绘制系统

简洁实用的Venn图绘制工具，支持10个预设主题。

## 特点

- ✅ 简单易用的API
- ✅ 13个预设主题（含3个线型主题）
- ✅ 4个白色边框主题
- ✅ 支持虚线、点线、点划线等边框样式
- ✅ 35+个配置参数
- ✅ 代码简洁清晰（370行）

## 快速开始

### 基础使用

```python
from venn_plot import VennPlot

venn = VennPlot()
venn.load_data('geometric_data_v3.pkl', 'plotdata_overlaps.csv')
venn.plot(title="我的Venn图")
```

### 使用预设主题

```python
from venn_plot import VennPlot
from plot_config import PlotStyle

# 13个预设主题
venn = VennPlot(PlotStyle.soft())     # 柔和
venn = VennPlot(PlotStyle.bold())     # 醒目
venn = VennPlot(PlotStyle.paper())    # 论文
venn = VennPlot(PlotStyle.white())    # 白边
venn = VennPlot(PlotStyle.clean())    # 清爽
venn = VennPlot(PlotStyle.dark())     # 深色
venn = VennPlot(PlotStyle.pastel())   # 粉彩
venn = VennPlot(PlotStyle.vivid())    # 鲜艳
venn = VennPlot(PlotStyle.poster())   # 海报
venn = VennPlot(PlotStyle.print())    # 打印
venn = VennPlot(PlotStyle.dashed())   # 虚线
venn = VennPlot(PlotStyle.dotted())   # 点线
venn = VennPlot(PlotStyle.dashdot())  # 点划线

venn.load_data('geo.pkl', 'overlap.csv')
venn.plot()
```

## 13个主题

| 主题 | 说明 | 边框 | 线型 | 用途 |
|------|------|------|------|------|
| soft | 柔和 | 灰色 | 实线 | 日常查看 |
| bold | 醒目 | 黑色粗 | 实线 | 演示展示 |
| paper | 论文 | 黑色 | 实线 | 学术论文 |
| white | 白边 | 白色 | 实线 | 深色背景 |
| clean | 清爽 | 白色 | 实线 | 清新风格 |
| dark | 深色 | 白色 | 实线 | 深色模式 |
| pastel | 粉彩 | 白色 | 实线 | 温和风格 |
| vivid | 鲜艳 | 黑色 | 实线 | 醒目展示 |
| poster | 海报 | 黑色粗 | 实线 | PPT演示 |
| print | 打印 | 黑色细 | 实线 | 高质量打印 |
| dashed | 虚线 | 黑色 | 虚线 | 区分效果 |
| dotted | 点线 | 灰色 | 点线 | 柔和区分 |
| dashdot | 点划线 | 深蓝 | 点划线 | 特殊标记 |

## 白色边框主题

4个白色边框主题：

```python
PlotStyle.white()    # 白边框+白标签
PlotStyle.clean()    # 白边框+黑标签
PlotStyle.dark()     # 白边框+深色背景
PlotStyle.pastel()   # 白边框+柔和配色
```

## 自定义样式

```python
from plot_config import PlotStyle

# 方式1: 创建自定义样式
style = PlotStyle(
    figsize=(12, 10),
    colormap='Set3',
    fill_alpha=0.5,
    edge_color='navy',
    edge_width=2.5,
    edge_style='--'  # 虚线边框
)

venn = VennPlot(style)
venn.load_data('geo.pkl', 'overlap.csv')
venn.plot()

# 方式2: 基于主题修改
style = PlotStyle.white()
style.edge_width = 4.0
style.edge_style = ':'  # 改为点线
style.fill_alpha = 0.8

venn = VennPlot(style)
```

### 可用的线型

```python
edge_style = '-'      # 实线（默认）
edge_style = '--'     # 虚线
edge_style = ':'      # 点线
edge_style = '-.'     # 点划线
edge_style = (0, (5, 5))  # 自定义：5像素线，5像素间隔
edge_style = (0, (3, 1, 1, 1))  # 自定义：复杂模式
```

## 高级功能

### 自定义颜色

```python
venn = VennPlot()
venn.load_data('geo.pkl', 'overlap.csv')

custom_colors = {
    'region1': '#FF6B6B',
    'region2': '#4ECDC4'
}
venn.set_custom_colors(custom_colors)
venn.plot()
```

### 自定义标签格式

```python
def format_label(value):
    return f"{int(value):,}"  # 添加千位分隔符

venn = VennPlot()
venn.load_data('geo.pkl', 'overlap.csv')
venn.set_label_formatter(format_label)
venn.plot()
```

### 保存图形

```python
venn = VennPlot()
venn.load_data('geo.pkl', 'overlap.csv')

venn.draw(title="My Venn")
venn.save('output.png', dpi=300)
venn.show()
venn.close()
```

### 获取统计信息

```python
venn = VennPlot()
venn.load_data('geo.pkl', 'overlap.csv')

stats = venn.get_statistics()
print(stats)
# {'n_regions': 4, 'n_sets': 4, 'total_size': 100, ...}
```

### 批量生成

```python
style = PlotStyle.paper()

for key in ['shape403', 'shape404']:
    venn = VennPlot(style)
    venn.load_data('geo.pkl', 'overlap.csv', key)
    venn.draw(title=f"Venn - {key}")
    venn.save(f'output_{key}.png')
    venn.close()
```

## 配置参数

PlotStyle 支持35+个配置参数：

```python
PlotStyle(
    # 图形设置
    figsize=(10, 10),
    dpi=100,
    facecolor='white',
    
    # 区域样式
    fill_alpha=0.4,
    colormap='viridis',
    
    # 边框样式
    edge_color='black',
    edge_width=2.0,
    edge_style='-',      # 线型: '-' 实线, '--' 虚线, ':' 点线, '-.' 点划线
    edge_alpha=1.0,
    
    # 标签样式
    label_fontsize=10,
    label_color='black',
    label_weight='bold',
    
    # 集合标签
    set_label_fontsize=12,
    set_label_color='darkred',
    
    # 标题
    title_fontsize=14,
    title_color='black',
    
    # 更多参数...
)
```

## 文件结构

```
refactored/
├── plot_config.py    # 样式配置（70行）
├── base_plot.py      # 基础绘图类（130行）
├── venn_plot.py      # Venn图实现（200行）
├── showcase.py       # 主题展示
├── example.py        # 使用示例
├── test_themes.py    # 测试脚本
├── THEMES.md         # 主题参考
└── QUICK_START.md    # 快速开始
```

## 示例

查看 `example.py` 获取8个完整示例：

1. 基础使用
2. 自定义样式
3. 预设样式
4. 自定义颜色
5. 自定义标签格式
6. 保存图形
7. 统计信息
8. 批量生成

运行示例：
```bash
cd refactored
python example.py
```

## 主题展示

运行主题展示：
```bash
cd refactored
python showcase.py
```

选择展示模式：
1. 展示所有主题
2. 展示白色边框主题
3. 按类别展示
4. 对比所有主题
5. 快速演示
6. 批量生成

## 测试

运行测试：
```bash
cd refactored
python test_themes.py
```

## 文档

- `THEMES.md` - 完整主题参考
- `QUICK_START.md` - 快速开始指南
- `FINAL_UPDATE.md` - 更新说明

## API参考

### VennPlot 类

```python
# 初始化
VennPlot(style=None)

# 数据加载
load_geometric_data(filepath, shape_key='shape404')
load_overlap_data(filepath)
load_data(geo_file, overlap_file, shape_key='shape404')

# 自定义
set_custom_colors(colors: Dict[str, str])
set_label_formatter(formatter: Callable)

# 绘制
draw(title="Venn Diagram", show_region_labels=True, show_set_labels=True)
plot(title="Venn Diagram", **kwargs)  # draw + show

# 其他
show()
save(filepath, dpi=None)
close()
get_statistics()
```

### PlotStyle 类

```python
# 创建
PlotStyle(...)           # 自定义参数
PlotStyle.soft()         # 柔和
PlotStyle.bold()         # 醒目
PlotStyle.paper()        # 论文
PlotStyle.white()        # 白边
PlotStyle.clean()        # 清爽
PlotStyle.dark()         # 深色
PlotStyle.pastel()       # 粉彩
PlotStyle.vivid()        # 鲜艳
PlotStyle.poster()       # 海报
PlotStyle.print()        # 打印
PlotStyle.dashed()       # 虚线
PlotStyle.dotted()       # 点线
PlotStyle.dashdot()      # 点划线
```

## 总结

这个简化版提供了：
- ✅ 简洁的代码（370行）
- ✅ 13个预设主题（含3个线型主题）
- ✅ 4个白色边框主题
- ✅ 支持虚线、点线、点划线等边框样式
- ✅ 35+个配置参数
- ✅ 完整的文档和示例

适合实际项目使用，代码清晰专业。
