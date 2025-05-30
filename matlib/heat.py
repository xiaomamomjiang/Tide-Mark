import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 12
})
# Set random seed for reproducibility
np.random.seed(42)

# Simulate PPO max action probabilities for 10 boundary nodes over 100 episodes
num_nodes = 10
num_episodes = 100

# Simulate gradual confidence increase
data = np.zeros((num_nodes, num_episodes))
for i in range(num_nodes):
    base = np.linspace(0.3, 0.95, num_episodes) + np.random.normal(0, 0.05, num_episodes)
    base = np.clip(base, 0.25, 1.0)
    data[i] = base

# Plot
plt.figure(figsize=(12, 6))
sns.heatmap(data, cmap="YlOrRd", cbar_kws={'label': 'Max Action Probability'}, xticklabels=10, yticklabels=1)
# plt.title('PPO Action Probabilities Across Episodes (10 Boundary Nodes)')
plt.xlabel('Episode')
plt.ylabel('Boundary Node Index')
plt.tight_layout()
plt.savefig("ppo_action_probabilities_heatmap_enhanced.pdf")
plt.show()
