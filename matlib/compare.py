import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

labels = ['Static Louvain', 'TGN + Louvain', 'DySAT + Louvain', 'TIDE-MARK (Ours)']
x = np.arange(len(labels))

# === 数据部分：填写你自己每个方法的五次实验数据 ===
modularity_data = [
    [0.531, 0.537, 0.534, 0.528, 0.533],
    [0.575, 0.578, 0.579, 0.574, 0.579],
    [0.588, 0.590, 0.593, 0.589, 0.595],
    [0.612, 0.617, 0.619, 0.610, 0.617]
]

conductance_avg = [0.282, 0.241, 0.228, 0.198]
ari_avg = [0.410, 0.613, 0.694, 0.742]
runtime_avg = [0.84, 1.75, 3.62, 2.37]

# === 函数：计算均值和 CI ===
def compute_ci(data_list):
    arr = np.array(data_list)
    mean = np.mean(arr)
    ci = stats.t.interval(0.95, len(arr)-1, loc=mean, scale=stats.sem(arr))
    lower = mean - ci[0]
    upper = ci[1] - mean
    return mean, lower, upper

# === Modularity CI 处理 ===
mod_means, mod_lowers, mod_uppers = [], [], []
for row in modularity_data:
    mean, l, u = compute_ci(row)
    mod_means.append(mean)
    mod_lowers.append(l)
    mod_uppers.append(u)

# === 开始绘图 ===
plt.rcParams.update({
    'font.size': 10,
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.linestyle': '--'
})

fig, axs = plt.subplots(2, 2)

# (a) Modularity 有置信区间
axs[0, 0].bar(x, mod_means, yerr=[mod_lowers, mod_uppers], capsize=5, color='green')
axs[0, 0].set_title('(a) Modularity $Q$ (Higher is better)')
axs[0, 0].set_ylabel('Modularity')
axs[0, 0].set_xticks(x)
axs[0, 0].set_xticklabels(labels, rotation=20)

# (b) Conductance 平均值
axs[0, 1].bar(x, conductance_avg, width=0.6, color='purple')
axs[0, 1].set_title('(b) Conductance $\\Phi$ (Lower is better)')
axs[0, 1].set_ylabel('Conductance')
axs[0, 1].set_xticks(x)
axs[0, 1].set_xticklabels(labels, rotation=20)

# (c) Temporal ARI 平均值
axs[1, 0].bar(x, ari_avg, width=0.6, color='royalblue')
axs[1, 0].set_title('(c) Temporal ARI (Higher is better)')
axs[1, 0].set_ylabel('ARI')
axs[1, 0].set_xticks(x)
axs[1, 0].set_xticklabels(labels, rotation=20)

# (d) Runtime 平均值
axs[1, 1].bar(x, runtime_avg, width=0.6, color='darkorange')
axs[1, 1].set_title('(d) Runtime (s) (Lower is better)')
axs[1, 1].set_ylabel('Seconds')
axs[1, 1].set_xticks(x)
axs[1, 1].set_xticklabels(labels, rotation=20)

# 统一网格样式
for ax in axs.flat:
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

plt.tight_layout()
plt.savefig("metrics_modularity_with_ci_only.pdf", bbox_inches='tight')
plt.show()
