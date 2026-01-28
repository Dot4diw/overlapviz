# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv('5set.csv')

# print(df.head())

# # df = pd.read_csv('your_data.csv')  # 包含所有集合的数据
# fig, ax = plt.subplots(figsize=(8, 8))
# for name, group in df.groupby('id'):
#     ax.plot(group['X'], group['Y'], linewidth=2)
# ax.set_aspect('equal')


# plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
df_shape = pd.read_csv('5set.csv')        # 形状线条数据
df_label = pd.read_csv('5set_label.csv')  # 标签坐标数据

fig, ax = plt.subplots(figsize=(10, 10))

# 绘制 5 个集合的轮廓
for name, group in df_shape.groupby('id'):
    ax.fill(group['X'], group['Y'], alpha=0.3, label=f"Fill {name}")
    ax.plot(group['X'], group['Y'], linewidth=2, label=f"Set {name}")

# 简单一行循环：添加标签
for _, row in df_label.iterrows():
    ax.text(
        row['X'], row['Y'], 
        str(row['id']), 
        fontsize=10, 
        ha='center', 
        va='center', 
        fontweight='bold'
    )

# 4. 图表修饰
ax.set_aspect('equal')
ax.set_axis_off()  # 隐藏坐标轴刻度，看起来更像维恩图
plt.title("5-Set Venn Diagram with Labels")
plt.show()
