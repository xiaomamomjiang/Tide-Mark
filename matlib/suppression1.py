import numpy as np
import matplotlib.pyplot as plt

# ✅ 设置统一风格，与前面图一致
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12
})

# 数据
categories = ['Modularity $Q$', 'Conductance $\Phi$', 'LCC Size']
original_values = [0.615, 0.198, 1.00]
suppressed_values = [0.574, 0.229, 0.786]

# 颜色（与前面一致）
colors = {
    "modularity": "#1b9e77",   # green
    "conductance": "#984ea3",  # purple
    "lcc": "#ff7f00"           # orange
}
bar_colors = [colors["modularity"], colors["conductance"], colors["lcc"]]

# 宽度和位置
bar_width = 0.35
x = np.arange(len(categories))

fig, ax = plt.subplots(figsize=(8, 5))

# 柱状图
ax.bar(x - bar_width/2, original_values, width=bar_width, label='Original (colored)',
       color=bar_colors, edgecolor='none')

ax.bar(x + bar_width/2, suppressed_values, width=bar_width, label='Suppressed (white + hatch)',
       color='none', edgecolor=bar_colors, hatch='///', linewidth=1.2)

# 设置
ax.set_ylabel('Score / Normalized Size')
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=0)
# ax.set_ylim(0, 1.1)  # 上方留白
# ax.grid(True, linestyle='--', alpha=0.4, axis='y')
ax.legend()


# # ✅ 图例移至顶部
# fig.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.05))

# ✅ 紧凑布局，顶部留空间
plt.tight_layout()
plt.savefig("suppression_simulation_consistent.pdf", bbox_inches='tight')
plt.show()
