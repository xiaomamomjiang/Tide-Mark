import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# 更新 Matplotlib 参数
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 12
})

# 设置方法、数据集、颜色
methods = ['Static Louvain', 'TGN + Louvain', 'DySAT + Louvain', 'TIDE-MARK']
datasets = ['PolitiFact', 'GossipCop', 'ReCOVery']
colors = ['green', 'blue', 'orange']

# 数据
modularity_means = [[0.534, 0.511, 0.497],
                    [0.553, 0.548, 0.534],
                    [0.596, 0.568, 0.563],
                    [0.617, 0.607, 0.602]]
modularity_cis = [[(0.522, 0.546), (0.500, 0.522), (0.484, 0.510)],
                  [(0.542, 0.564), (0.536, 0.560), (0.522, 0.546)],
                  [(0.584, 0.608), (0.555, 0.581), (0.552, 0.574)],
                  [(0.603, 0.631), (0.595, 0.619), (0.587, 0.617)]]
conductance_means = [[0.321, 0.338, 0.361],
                     [0.312, 0.319, 0.339],
                     [0.300, 0.301, 0.326],
                     [0.287, 0.291, 0.299]]
conductance_cis = [[(0.311, 0.331), (0.326, 0.350), (0.348, 0.374)],
                   [(0.301, 0.323), (0.308, 0.330), (0.327, 0.351)],
                   [(0.290, 0.310), (0.291, 0.311), (0.312, 0.340)],
                   [(0.276, 0.298), (0.281, 0.301), (0.286, 0.312)]]
ari_means = [[0.624, 0.600, 0.583],
             [0.636, 0.632, 0.612],
             [0.698, 0.681, 0.662],
             [0.758, 0.734, 0.727]]
ari_cis = [[(0.610, 0.638), (0.588, 0.612), (0.570, 0.596)],
           [(0.624, 0.648), (0.620, 0.644), (0.600, 0.624)],
           [(0.684, 0.712), (0.667, 0.695), (0.648, 0.676)],
           [(0.744, 0.772), (0.720, 0.748), (0.715, 0.739)]]
runtime_means = [[0.91, 0.88, 0.95],
                 [1.75, 1.69, 1.82],
                 [3.58, 3.45, 3.66],
                 [2.43, 2.35, 2.48]]
runtime_cis = [[(0.89, 0.93), (0.86, 0.90), (0.93, 0.97)],
               [(1.72, 1.78), (1.66, 1.72), (1.79, 1.85)],
               [(3.55, 3.61), (3.42, 3.48), (3.63, 3.69)],
               [(2.40, 2.46), (2.32, 2.38), (2.45, 2.51)]]

# 准备绘图
x = np.arange(len(methods))
width = 0.2
fig, axs = plt.subplots(2, 2, figsize=(14, 8))
axs = axs.flatten()

# (a) Modularity
for i, (dataset, color) in enumerate(zip(datasets, colors)):
    means = [modularity_means[m][i] for m in range(len(methods))]
    lowers = [means[j] - modularity_cis[j][i][0] for j in range(len(methods))]
    uppers = [modularity_cis[j][i][1] - means[j] for j in range(len(methods))]
    axs[0].bar(x + (i - 1) * width, means, width, label=dataset, color=color, yerr=[lowers, uppers], capsize=5)
axs[0].set_title("(a) Modularity $Q$")
axs[0].set_ylabel("Modularity")
axs[0].set_xticks(x)
axs[0].set_xticklabels(methods, rotation=15)

# (b) Conductance
for i, color in enumerate(colors):
    means = [conductance_means[m][i] for m in range(len(methods))]
    lowers = [means[j] - conductance_cis[j][i][0] for j in range(len(methods))]
    uppers = [conductance_cis[j][i][1] - means[j] for j in range(len(methods))]
    axs[1].bar(x + (i - 1) * width, means, width, color=color, yerr=[lowers, uppers], capsize=5)
axs[1].set_title("(b) Conductance $\Phi$")
axs[1].set_ylabel("Conductance")
axs[1].set_xticks(x)
axs[1].set_xticklabels(methods, rotation=15)

# (c) ARI with CI
for i, color in enumerate(colors):
    means = [ari_means[m][i] for m in range(len(methods))]
    lowers = [means[j] - ari_cis[j][i][0] for j in range(len(methods))]
    uppers = [ari_cis[j][i][1] - means[j] for j in range(len(methods))]
    axs[2].bar(x + (i - 1) * width, means, width, color=color, yerr=[lowers, uppers], capsize=5)
axs[2].set_title("(c) Temporal ARI")
axs[2].set_ylabel("ARI")
axs[2].set_xticks(x)
axs[2].set_xticklabels(methods, rotation=15)

# (d) Runtime with CI
for i, color in enumerate(colors):
    means = [runtime_means[m][i] for m in range(len(methods))]
    lowers = [means[j] - runtime_cis[j][i][0] for j in range(len(methods))]
    uppers = [runtime_cis[j][i][1] - means[j] for j in range(len(methods))]
    axs[3].bar(x + (i - 1) * width, means, width, color=color, yerr=[lowers, uppers], capsize=5)
axs[3].set_title("(d) Runtime per snapshot")
axs[3].set_ylabel("Seconds")
axs[3].set_xticks(x)
axs[3].set_xticklabels(methods, rotation=15)

# 图例
legend_elements = [Patch(facecolor=colors[i], label=datasets[i]) for i in range(len(datasets))]
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize='medium', bbox_to_anchor=(0.5, 1.0))

# import ace_tools as tools; tools.display_dataframe_to_user(name="Updated Metric Plot", dataframe=None)
plt.savefig("grouped_metrics_plot.pdf", bbox_inches='tight')
plt.show()