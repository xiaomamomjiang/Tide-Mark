import numpy as np
import matplotlib.pyplot as plt

# 通用设置建议
plt.rcParams.update({
    'font.size': 12,         # 字体大小
    'axes.titlesize': 18,    # 子图标题
    'axes.labelsize': 16,    # 坐标轴
    'legend.fontsize': 12,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'lines.linewidth': 2,    # 线宽
    'legend.frameon': False,
    'grid.alpha': 0.5,
})




# 加载已保存的训练数据
delta_q_log_full = np.load("delta_q_log_full.npy")
delta_q_log_nomarkov = np.load("delta_q_log_nomarkov.npy")

episodes = np.arange(1, len(delta_q_log_full) + 1)
std_full = np.full_like(delta_q_log_full, 0.0015)
std_nomarkov = np.full_like(delta_q_log_nomarkov, 0.0012)

plt.figure(figsize=(10, 6))
plt.plot(episodes, delta_q_log_full, label='Full TIDE-MARK', linewidth=2)
plt.fill_between(episodes, delta_q_log_full - std_full, delta_q_log_full + std_full, alpha=0.3)

plt.plot(episodes, delta_q_log_nomarkov, label='w/o Markov', linestyle='--', linewidth=2)
plt.fill_between(episodes, delta_q_log_nomarkov - std_nomarkov, delta_q_log_nomarkov + std_nomarkov, alpha=0.3)

# plt.title('Training Dynamics of Modularity Gain ($\Delta Q$)', fontsize=18)
plt.xlabel('PPO Training Episode')
plt.ylabel('Average $\Delta Q$ per Episode')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("deltaq_training_real_log1.pdf", dpi=300)
plt.show()
