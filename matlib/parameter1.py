import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 12
})


x_alpha = np.linspace(0.1, 1.0, 10)
x_beta = np.linspace(0.1, 1.0, 10)
x_k = np.linspace(5, 50, 10, dtype=int)
x_eta = np.array([1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 7e-3, 9e-3, 1e-2, 2e-2])


mod_alpha = {
    "PolitiFact": np.array([0.591, 0.600, 0.615, 0.612, 0.607, 0.606, 0.599, 0.590, 0.582, 0.575]),
    "GossipCop": np.array([0.583, 0.590, 0.604, 0.601, 0.597, 0.595, 0.591, 0.583, 0.576, 0.570]),
    "ReCOVery":  np.array([0.581, 0.588, 0.602, 0.598, 0.595, 0.594, 0.589, 0.581, 0.574, 0.567])
}
mod_alpha_ci = {
    "PolitiFact": np.array([0.008, 0.007, 0.009, 0.010, 0.009, 0.008, 0.007, 0.006, 0.008, 0.009]),
    "GossipCop": np.array([0.009, 0.008, 0.010, 0.010, 0.009, 0.008, 0.007, 0.007, 0.008, 0.009]),
    "ReCOVery":  np.array([0.009, 0.009, 0.011, 0.010, 0.009, 0.009, 0.008, 0.008, 0.009, 0.009])
}

ari_beta = {
    "PolitiFact": np.array([0.705, 0.715, 0.729, 0.742, 0.740, 0.735, 0.728, 0.720, 0.710, 0.700]),
    "GossipCop": np.array([0.692, 0.702, 0.715, 0.728, 0.726, 0.722, 0.715, 0.707, 0.698, 0.687]),
    "ReCOVery":  np.array([0.683, 0.692, 0.705, 0.717, 0.715, 0.710, 0.703, 0.695, 0.687, 0.675])
}
ari_beta_ci = {
    "PolitiFact": np.array([0.006, 0.006, 0.006, 0.007, 0.006, 0.006, 0.006, 0.006, 0.006, 0.007]),
    "GossipCop": np.array([0.007, 0.007, 0.008, 0.008, 0.007, 0.007, 0.006, 0.007, 0.007, 0.008]),
    "ReCOVery":  np.array([0.007, 0.007, 0.008, 0.008, 0.007, 0.007, 0.006, 0.007, 0.007, 0.008])
}

mod_k = {
    "PolitiFact": np.array([0.594, 0.603, 0.615, 0.617, 0.616, 0.614, 0.610, 0.610, 0.606, 0.602]),
    "GossipCop": np.array([0.586, 0.595, 0.607, 0.609, 0.608, 0.606, 0.602, 0.602, 0.599, 0.595]),
    "ReCOVery":  np.array([0.582, 0.591, 0.603, 0.605, 0.604, 0.602, 0.598, 0.598, 0.595, 0.591])
}
mod_k_ci = {
    "PolitiFact": np.array([0.007, 0.007, 0.008, 0.009, 0.008, 0.008, 0.007, 0.007, 0.007, 0.007]),
    "GossipCop": np.array([0.008, 0.008, 0.009, 0.009, 0.008, 0.008, 0.007, 0.007, 0.008, 0.008]),
    "ReCOVery":  np.array([0.008, 0.008, 0.010, 0.009, 0.008, 0.008, 0.007, 0.007, 0.008, 0.008])
}

ari_eta = {
    "PolitiFact": np.array([0.735, 0.738, 0.741, 0.742, 0.740, 0.735, 0.728, 0.720, 0.715, 0.700]),
    "GossipCop": np.array([0.721, 0.725, 0.728, 0.729, 0.727, 0.723, 0.716, 0.708, 0.702, 0.687]),
    "ReCOVery":  np.array([0.710, 0.715, 0.719, 0.720, 0.718, 0.713, 0.707, 0.700, 0.694, 0.678])
}
ari_eta_ci = {
    "PolitiFact": np.array([0.005, 0.005, 0.005, 0.006, 0.005, 0.005, 0.005, 0.005, 0.005, 0.006]),
    "GossipCop": np.array([0.006, 0.006, 0.007, 0.007, 0.006, 0.006, 0.006, 0.006, 0.006, 0.007]),
    "ReCOVery":  np.array([0.006, 0.006, 0.007, 0.007, 0.006, 0.006, 0.006, 0.006, 0.006, 0.007])
}

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
datasets = ["PolitiFact", "GossipCop", "ReCOVery"]

# (a) alpha - modularity
for ds in datasets:
    axs[0, 0].errorbar(x_alpha, mod_alpha[ds], yerr=mod_alpha_ci[ds], fmt='-o', capsize=4, label=ds)
axs[0, 0].set_title('(a) Varying $\\alpha$')
axs[0, 0].set_xlabel('$\\alpha$')
axs[0, 0].set_ylabel('Modularity')
axs[0, 0].grid(True, linestyle='--', alpha=0.5)

# (b) beta - ARI
for ds in datasets:
    axs[0, 1].errorbar(x_beta, ari_beta[ds], yerr=ari_beta_ci[ds], fmt='-o', capsize=4, label=ds)
axs[0, 1].set_title('(b) Varying $\\beta$')
axs[0, 1].set_xlabel('$\\beta$')
axs[0, 1].set_ylabel('Temporal ARI')
axs[0, 1].grid(True, linestyle='--', alpha=0.5)

# (c) k - modularity
for ds in datasets:
    axs[1, 0].errorbar(x_k, mod_k[ds], yerr=mod_k_ci[ds], fmt='-o', capsize=4, label=ds)
axs[1, 0].set_title('(c) Varying $k$')
axs[1, 0].set_xlabel('$k$')
axs[1, 0].set_ylabel('Modularity')
axs[1, 0].grid(True, linestyle='--', alpha=0.5)

# (d) eta - ARI
for ds in datasets:
    axs[1, 1].errorbar(x_eta, ari_eta[ds], yerr=ari_eta_ci[ds], fmt='-o', capsize=4, label=ds)
axs[1, 1].set_title('(d) Varying PPO learning rate $\\eta$ (log scale)')
axs[1, 1].set_xlabel('$\\eta$')
axs[1, 1].set_ylabel('Temporal ARI')
axs[1, 1].set_xscale('log')
axs[1, 1].grid(True, linestyle='--', alpha=0.5)

fig.legend(datasets, loc='upper center', ncol=3, fontsize=12, bbox_to_anchor=(0.5, 1.0))
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("parameter_sensitivity_by_dataset_top_legend.pdf")
plt.show()
