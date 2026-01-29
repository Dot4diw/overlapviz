import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import to_rgba
import matplotlib.cm as cm

with open('geometric_data2.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

fig, ax = plt.subplots(figsize=(10, 10))

# 提取数据
df_edges = loaded_dict['shape301']['set_edge']
set_label = loaded_dict['shape301']['set_label']
df_label = loaded_dict['shape301']['region_label']

# 1. 计算全局边界，用于后续设置坐标轴范围
x_min, x_max = df_edges['X'].min(), df_edges['X'].max()
y_min, y_max = df_edges['Y'].min(), df_edges['Y'].max()

# 第一步：只画所有区域的填充 (无边框) ---
for name, group in df_edges.groupby('id'):
    verts = group[['X', 'Y']].values
    colors = [cm.viridis(i / len(df_edges.groupby('id'))) for i in range(len(df_edges.groupby('id')))]
    poly = Polygon(verts, 
                   closed=True, 
                   facecolor=to_rgba(colors[int(name) % len(colors)], 0.4), 
                   edgecolor='none',  # 先不画边框
                   zorder=1)          # 放在底层
    ax.add_patch(poly)

# 第二步：统一画所有区域的边框 (无填充) ---
for name, group in df_edges.groupby('id'):
    verts = group[['X', 'Y']].values
    # 强制不使用 Polygon 的自动闭合，或者确保边缘完全一致
    # 使用 plot 绘制可以更精准地控制线型
    ax.plot(group['X'], group['Y'], 
            color='black',            # 纯黑
            linewidth=2, 
            alpha=1.0,                # 绝对不透明
            zorder=2) # # 放在顶层图层

# 第三步：添加区域标签 (Region Labels)
for _, row in df_label.iterrows():
    ax.text(row['X'], row['Y'], str(row['id']), 
            fontsize=10, ha='center', va='center', fontweight='bold')

# 第四步：添加集合标签 (Set Labels)
for _, row in set_label.iterrows():
    ax.text(row['X'], row['Y'], str(row['id']), 
            fontsize=12, ha='center', va='center', fontweight='bold', color='darkred')

# # 使用列表存储 (dataframe, fontsize, color) 元组
# for df, fs, col in [(df_label, 10, 'black'), (set_label, 12, 'darkred')]:
#     for _, row in df.iterrows():
#         ax.text(row['X'], row['Y'], str(row['id']),
#                 fontsize=fs, color=col,
#                 ha='center', va='center', fontweight='bold')

# 设置坐标轴范围，添加 10% 的填充空间 (Padding)
padding_x = (x_max - x_min) * 0.1
padding_y = (y_max - y_min) * 0.1

ax.set_xlim(x_min - padding_x, x_max + padding_x)
ax.set_ylim(y_min - padding_y, y_max + padding_y)

# 3. 样式优化
ax.set_aspect('equal')
ax.set_axis_off() 
plt.title("Polygon Based Venn Diagram (Fixed Limits)", fontsize=14)

plt.show()
