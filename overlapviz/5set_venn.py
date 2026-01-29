

import pickle
import pandas as pd
import matplotlib.pyplot as plt

#读取预定义的shape数据
with open('geometric_data.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)


fig, ax = plt.subplots(figsize=(10, 10))

df_edges = loaded_dict['shape404']['set_edge']
set_label = loaded_dict['shape404']['set_label']
df_label = loaded_dict['shape404']['region_label']

#绘制shape
for name, group in df_edges.groupby('id'):
    ax.fill(group['X'], group['Y'], alpha=0.4, label=f"Fill {name}")
    ax.plot(group['X'], group['Y'], linewidth=2, label=f"Set {name}")

# 添加区域标签
for _, row in df_label.iterrows():
    ax.text(
        row['X'], row['Y'], 
        str(row['id']), 
        fontsize=10, 
        ha='center', 
        va='center', 
        fontweight='bold'
    )

# 添加集合标签
for _, row in set_label.iterrows():
    ax.text(
        row['X'], row['Y'], 
        str(row['id']), 
        fontsize=10, 
        ha='center', 
        va='center', 
        fontweight='bold'
    )


ax.set_aspect('equal')
ax.set_axis_off()  # 隐藏坐标轴刻度，看起来更像维恩图
plt.title("2-Set Venn Diagram with Labels")
plt.show()
