import matplotlib.pyplot as plt
import numpy as np



plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 12
})


metrics = ['Modularity $Q$', 'Conductance $\Phi$', 'Temporal ARI']

data = {
    'PolitiFact': {
        'fake': [0.617, 0.287, 0.758],
        'real': [0.547, 0.371, 0.661],
        'fake_err': [0.014, 0.014, 0.016],
        'real_err': [0.016, 0.014, 0.016],
    },
    'GossipCop': {
        'fake': [0.607, 0.291, 0.734],
        'real': [0.552, 0.378, 0.659],
        'fake_err': [0.013, 0.013, 0.016],
        'real_err': [0.014, 0.013, 0.016],
    },
    'ReCOVery': {
        'fake': [0.602, 0.299, 0.727],
        'real': [0.537, 0.373, 0.645],
        'fake_err': [0.013, 0.013, 0.016],
        'real_err': [0.014, 0.013, 0.016],
    }
}

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for idx, (dataset, values) in enumerate(data.items()):
    x = np.arange(len(metrics))
    width = 0.35
    ax = axs[idx]
    ax.bar(x - width/2, values['fake'], width, yerr=values['fake_err'], label='Fake', capsize=4)
    ax.bar(x + width/2, values['real'], width, yerr=values['real_err'], label='Real', capsize=4)
    ax.set_title(f"({chr(97 + idx)}) {dataset}")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha='right')
    if idx == 0:
        ax.set_ylabel('Metric Value')
    ax.set_ylim(0.2, 0.8)

# fig.suptitle('Structural Differences between Fake and Real News')

fig.legend(['Fake', 'Real'], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.01))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("barplot_fake_vs_real_per_dataset.pdf", bbox_inches='tight')
plt.show()
