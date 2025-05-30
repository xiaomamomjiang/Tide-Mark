import matplotlib.pyplot as plt
import numpy as np

# Snapshot indices
t = np.arange(1, 6)

# 假设我们选中 boundary user 的社区编号随时间的变化
tidemark = [1, 1, 1, 2, 2]
wo_rl = [1, 2, 3, 2, 3]
wo_markov = [1, 3, 1, 3, 2]

# 绘图
plt.figure(figsize=(7, 4))
plt.plot(t, tidemark, 'o-', label='TIDE-MARK', linewidth=2)
plt.plot(t, wo_rl, 's--', label='w/o RL', linewidth=2)
plt.plot(t, wo_markov, '^--', label='w/o Markov', linewidth=2)

plt.xlabel('Snapshot Index', fontsize=12)
plt.ylabel('Community ID', fontsize=12)
plt.title('Community Trajectory of a Boundary User', fontsize=14)
plt.xticks(t)
plt.yticks([0, 1, 2, 3])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("boundary_user_trajectory.pdf")
plt.show()
