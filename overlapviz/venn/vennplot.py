# Import required libraries
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import to_rgba
import matplotlib.cm as cm

# Load geometric data from pickle file
with open('geometric_data_v2.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

# Create figure and axis with 10x10 inch size
fig, ax = plt.subplots(figsize=(10, 10))

# Extract data from loaded dictionary
df_edges = loaded_dict['shape404']['set_edge']
set_label = loaded_dict['shape404']['set_label']
df_label = loaded_dict['shape404']['region_label']

# Step 1: Calculate global boundaries for coordinate axis limits
x_min, x_max = df_edges['X'].min(), df_edges['X'].max()
y_min, y_max = df_edges['Y'].min(), df_edges['Y'].max()

# Step 2: Draw filled regions for all areas (no edges) ---
for name, group in df_edges.groupby('id'):
    verts = group[['X', 'Y']].values
    # Generate colors using viridis colormap
    colors = [cm.viridis(i / len(df_edges.groupby('id'))) for i in range(len(df_edges.groupby('id')))]
    poly = Polygon(verts, 
                   closed=True, 
                   facecolor=to_rgba(colors[int(name) % len(colors)], 0.4),  # 0.4 = transparency
                   edgecolor='none',  # No edge for now
                   zorder=1)          # Place in background
    ax.add_patch(poly)

# Step 3: Draw borders for all areas (no fill) ---
for name, group in df_edges.groupby('id'):
    verts = group[['X', 'Y']].values
    # Use plot instead of Polygon for precise control over line style
    # This ensures consistent edge rendering
    ax.plot(group['X'], group['Y'], 
            color='black',            # Pure black
            linewidth=2, 
            alpha=1.0,                # Completely opaque
            zorder=2)                 # Place above fill

# Step 4: Add region labels
for _, row in df_label.iterrows():
    ax.text(row['X'], row['Y'], str(row['id']), 
            fontsize=10, ha='center', va='center', fontweight='bold')

# Step 5: Add set labels
for _, row in set_label.iterrows():
    ax.text(row['X'], row['Y'], str(row['id']), 
            fontsize=12, ha='center', va='center', fontweight='bold', color='darkred')

# Alternative approach using list comprehension (commented out)
# for df, fs, col in [(df_label, 10, 'black'), (set_label, 12, 'darkred')]:
#     for _, row in df.iterrows():
#         ax.text(row['X'], row['Y'], str(row['id']),
#                 fontsize=fs, color=col,
#                 ha='center', va='center', fontweight='bold')

# Step 6: Set axis limits with 10% padding
padding_x = (x_max - x_min) * 0.1
padding_y = (y_max - y_min) * 0.1

ax.set_xlim(x_min - padding_x, x_max + padding_x)
ax.set_ylim(y_min - padding_y, y_max + padding_y)

# Step 7: Style optimization
ax.set_aspect('equal')    # Maintain 1:1 aspect ratio
ax.set_axis_off()        # Hide axis lines and labels
plt.title("Polygon Based Venn Diagram (Fixed Limits)", fontsize=14)

# Display the plot
plt.show()
