import matplotlib.pyplot as plt
import numpy as np

# 假设每个模型有100个episode的数据
episodes = np.arange(1, 101)

# 模拟ΔQ曲线（示意）
full = 0.01 * np.log1p(episodes) + 0.05
no_rl = 0.009 * np.log1p(episodes) + 0.02
no_markov = 0.008 * np.log1p(episodes) + 0.01
no_both = 0.007 * np.log1p(episodes)

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(episodes, full, label="TIDE-MARK (Full)", linewidth=2)
plt.plot(episodes, no_rl, label="w/o RL Refinement", linestyle="--")
plt.plot(episodes, no_markov, label="w/o Markov Transition", linestyle="-.")
plt.plot(episodes, no_both, label="w/o Both", linestyle=":")

plt.xlabel("Training Episode")
plt.ylabel("Average $\Delta Q$")
plt.title("Training Dynamics of Modularity Gain")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("deltaq_training.pdf", bbox_inches="tight")
plt.show()
