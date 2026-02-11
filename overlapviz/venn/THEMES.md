# 主题参考

## 13个预设主题

所有主题图形大小统一为 `(10, 10)`

### 基础主题（10个）

#### soft - 柔和
```python
PlotStyle.soft()
```
- 柔和Pastel配色
- 灰色细边框（实线）
- 低透明度
- 适合日常查看

#### bold - 醒目
```python
PlotStyle.bold()
```
- 鲜艳Set1配色
- 黑色粗边框（实线）
- 大字体
- 适合演示展示

#### paper - 论文
```python
PlotStyle.paper()
```
- 专业tab10配色
- 高分辨率300 DPI
- 黑色边框（实线）
- 适合学术论文

#### white - 白边
```python
PlotStyle.white()
```
- 白色边框（实线）
- 白色标签
- 适合深色背景

#### clean - 清爽
```python
PlotStyle.clean()
```
- 白色边框（实线）
- 黑色标签
- 清新风格

#### dark - 深色
```python
PlotStyle.dark()
```
- 深色背景
- 白色边框（实线）
- 白色标签
- 适合深色模式

#### pastel - 粉彩
```python
PlotStyle.pastel()
```
- 柔和Pastel2配色
- 白色边框（实线）
- 温和风格

#### vivid - 鲜艳
```python
PlotStyle.vivid()
```
- 高对比度bright配色
- 黑色边框（实线）
- 醒目展示

#### poster - 海报
```python
PlotStyle.poster()
```
- 大字体
- 黑色粗边框（实线）
- 适合PPT演示

#### print - 打印
```python
PlotStyle.print()
```
- 高DPI打印质量
- 黑色细边框（实线）
- 适合高质量打印

### 线型主题（3个）

#### dashed - 虚线
```python
PlotStyle.dashed()
```
- 黑色虚线边框（`--`）
- Set2配色
- 适合区分不同区域

#### dotted - 点线
```python
PlotStyle.dotted()
```
- 灰色点线边框（`:`）
- Pastel1配色
- 柔和的区分效果

#### dashdot - 点划线
```python
PlotStyle.dashdot()
```
- 深蓝点划线边框（`-.`）
- Set3配色
- 特殊标记效果

## 主题对比表

| 主题 | 边框颜色 | 线型 | 边框宽度 | 透明度 | DPI | 适用场景 |
|------|----------|------|----------|--------|-----|----------|
| soft | 灰色 | 实线 | 1.5 | 0.3 | 100 | 日常查看 |
| bold | 黑色 | 实线 | 3.0 | 0.6 | 100 | 演示展示 |
| paper | 黑色 | 实线 | 1.5 | 0.4 | 100 | 学术论文 |
| white | 白色 | 实线 | 2.5 | 0.7 | 100 | 深色背景 |
| clean | 白色 | 实线 | 3.0 | 0.5 | 100 | 清新风格 |
| dark | 白色 | 实线 | 2.0 | 0.6 | 100 | 深色模式 |
| pastel | 白色 | 实线 | 2.5 | 0.6 | 100 | 温和风格 |
| vivid | 黑色 | 实线 | 2.5 | 0.7 | 100 | 醒目展示 |
| poster | 黑色 | 实线 | 3.5 | 0.6 | 100 | PPT演示 |
| print | 黑色 | 实线 | 1.0 | 0.4 | 100 | 高质量打印 |
| dashed | 黑色 | 虚线 | 2.0 | 0.5 | 100 | 区分效果 |
| dotted | 灰色 | 点线 | 2.5 | 0.4 | 100 | 柔和区分 |
| dashdot | 深蓝 | 点划线 | 2.0 | 0.5 | 100 | 特殊标记 |

## 白色边框主题

4个白色边框主题：
- `white` - 白边框+白标签
- `clean` - 白边框+黑标签
- `dark` - 白边框+深色背景
- `pastel` - 白边框+柔和配色

## 线型说明

### 可用线型

```python
edge_style = '-'      # 实线（默认）
edge_style = '--'     # 虚线
edge_style = ':'      # 点线
edge_style = '-.'     # 点划线
```

### 自定义线型

```python
# 自定义虚线模式：(偏移, (线长, 间隔))
edge_style = (0, (5, 5))        # 5像素线，5像素间隔
edge_style = (0, (3, 1, 1, 1))  # 复杂模式：3线-1空-1线-1空
edge_style = (0, (10, 2))       # 10像素线，2像素间隔
```

### 使用示例

```python
from venn_plot import VennPlot
from plot_config import PlotStyle

# 使用预设虚线主题
venn = VennPlot(PlotStyle.dashed())
venn.load_data('geo.pkl', 'overlap.csv')
venn.plot()

# 自定义线型
style = PlotStyle.bold()
style.edge_style = ':'  # 改为点线
style.edge_width = 3.0

venn = VennPlot(style)
venn.load_data('geo.pkl', 'overlap.csv')
venn.plot()

# 完全自定义
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

## 主题选择建议

### 按用途选择

- **日常使用**: soft, clean
- **学术论文**: paper, print
- **演示展示**: bold, poster, vivid
- **深色背景**: white, dark
- **区分效果**: dashed, dotted, dashdot
- **温和风格**: pastel, soft

### 按边框选择

- **实线边框**: soft, bold, paper, white, clean, dark, pastel, vivid, poster, print
- **虚线边框**: dashed
- **点线边框**: dotted
- **点划线边框**: dashdot

### 按颜色选择

- **黑色边框**: soft, bold, paper, vivid, poster, print, dashed
- **白色边框**: white, clean, dark, pastel
- **灰色边框**: dotted
- **深蓝边框**: dashdot

## 自定义主题

基于现有主题修改：

```python
from plot_config import PlotStyle

# 基于white主题，改为虚线
style = PlotStyle.white()
style.edge_style = '--'
style.edge_width = 3.0

# 基于bold主题，改为点线
style = PlotStyle.bold()
style.edge_style = ':'
style.edge_color = 'navy'

# 基于clean主题，改为点划线
style = PlotStyle.clean()
style.edge_style = '-.'
style.edge_width = 2.5
```

## 完整配置参数

所有主题都支持以下参数：

```python
PlotStyle(
    # 图形设置
    figsize=(10, 10),
    dpi=100,
    facecolor='white',
    edgecolor='white',
    
    # 区域样式
    fill_alpha=0.4,
    colormap='viridis',
    
    # 边框样式
    edge_color='black',
    edge_width=2.0,
    edge_style='-',      # 线型
    edge_alpha=1.0,
    
    # 标签样式
    label_fontsize=10,
    label_color='black',
    label_weight='bold',
    label_alpha=1.0,
    label_ha='center',
    label_va='center',
    
    # 集合标签样式
    set_label_fontsize=12,
    set_label_color='darkred',
    set_label_weight='bold',
    set_label_alpha=1.0,
    set_label_ha='center',
    set_label_va='center',
    
    # 标题样式
    title_fontsize=14,
    title_weight='bold',
    title_color='black',
    title_pad=20.0,
    title_loc='center',
    
    # 坐标轴
    padding_ratio=0.1,
    show_axis=False,
    axis_color='black',
    
    # 背景和网格
    show_grid=False,
    grid_color='gray',
    grid_alpha=0.3,
    grid_linestyle='--'
)
```
