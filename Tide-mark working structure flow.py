import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Ellipse
import numpy as np

def rbox(ax, x, y, w, h, colour, label, fs=8):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                       fc=colour, ec="k", lw=1.1, zorder=3)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=fs, weight="semibold")
    return p

# ---------------- Layout ----------------
row_y, frame_h, frame_w, gap = 1.55, 0.9, 2.6, 0.4
text_x = 0.4
ppo_x   = text_x + frame_w + gap
graph_x = ppo_x  + frame_w + gap
split_x = graph_x + frame_w + 0.3

# Colours
C_GNN_LOUVAIN = "#81d4fa"
C_GNN_MARKOV  = "#a5d6a7"
C_CLASSIFIER  = "#ce93d8"
C_ROB, C_LR   = "#aed581", "#d1c4e9"
C_BAR1, C_BAR2 = "#ffb3b3", "#e0e0e0"
C_NODE, C_INSERT = "#ffdede", "#ff6e6e"

fig, ax = plt.subplots(figsize=(17, 8))
ax.axis("off")

# Divider
ax.plot([split_x, split_x], [0.3, 3.2], color="k", lw=1.4, ls="--")
ax.text(split_x - 1, 3.25, "Graph Processing", ha="right", fontsize=10, weight="bold", color="#8b0000")
ax.text(split_x + 1, 3.25, "Community Detection", ha="left", fontsize=10, weight="bold", style="italic", color="#2e7d32")

# ---------- Text Encoder ----------
ax.add_patch(Rectangle((text_x, row_y), frame_w, frame_h, fill=False, ls="--"))
ax.text(text_x + 0.05, row_y + frame_h + 0.05, "Pre-Processing", fontsize=9, weight="bold")
for i in range(3):
    ax.add_patch(Rectangle((text_x + 0.2 + 0.15*i, row_y + 0.35), 0.5, 0.25, fc="white", ec="k"))
pp_steps = ["Tokenise", "Clean", "Vectorise"]
for i, s in enumerate(pp_steps):
    rbox(ax, text_x + frame_w - 2.1 + i*0.7, row_y + 0.15, 0.6, 0.35, C_ROB, s, fs=6)

# ---------- Temporal Embedding ----------
ax.add_patch(Rectangle((ppo_x, row_y), frame_w, frame_h, fill=False, ls="--"))
ax.text(ppo_x + 0.10, row_y + frame_h + 0.05, "Temporal Embedding", fontsize=9, weight="bold")
circles = [(ppo_x + 0.5, row_y + 0.55),
           (ppo_x + 0.9, row_y + 0.7),
           (ppo_x + 1.2, row_y + 0.45)]
for (cx, cy) in circles:
    ax.add_patch(Circle((cx, cy), 0.09, fc=C_NODE, ec="k"))
ins_cx, ins_cy = ppo_x + 0.75, row_y + 0.6
ax.add_patch(Circle((ins_cx, ins_cy), 0.11, fc=C_INSERT, ec="k", lw=1.3))
ax.text(ins_cx, ins_cy - 0.17, "$inserted$", fontsize=6, ha="center", color=C_INSERT)
rbox(ax, ppo_x + frame_w - 1.0, row_y + 0.2, 0.9, 0.45, C_LR, "TGN")

# ---------- Clustering ----------
ax.add_patch(Rectangle((graph_x, row_y), frame_w, frame_h, fill=False, ls="--"))
ax.text(graph_x + 0.05, row_y + frame_h + 0.05, "Clustering", fontsize=9, weight="bold")
cl1 = [(graph_x + 0.5, row_y + 0.6),(graph_x + 0.7, row_y + 0.55),(graph_x + 0.55, row_y + 0.35)]
cl2 = [(graph_x + 1.3, row_y + 0.7),(graph_x + 1.45, row_y + 0.5),(graph_x + 1.25, row_y + 0.4)]
for (cx, cy) in cl1 + cl2:
    ax.add_patch(Circle((cx, cy), 0.1, fc=C_NODE, ec="k"))
ax.add_patch(Ellipse((graph_x + 0.6,  row_y + 0.5), 0.6,  0.5, angle=15,  fill=False, ls="--", lw=1.0, ec="#666"))
ax.add_patch(Ellipse((graph_x + 1.35, row_y + 0.55), 0.65, 0.55, angle=-10, fill=False, ls="--", lw=1.0, ec="#666"))
rbox(ax, graph_x + frame_w - 1.0, row_y + 0.2, 0.9, 0.45, C_GNN_LOUVAIN, "Louvain\nAlgorithm", fs=7)

# ---------- Arrows between sections ----------
centre_y = row_y + 0.45
arrow_kw = dict(arrowstyle="->", lw=1.1)
ax.annotate("", (ppo_x, centre_y), (text_x + frame_w, centre_y), arrowprops=arrow_kw)
ax.text((text_x + frame_w + ppo_x)/2, centre_y + 0.18, "$Text$", fontsize=7, ha="center")
ax.annotate("", (graph_x, centre_y), (ppo_x + frame_w, centre_y), arrowprops=arrow_kw)
ax.text((ppo_x + frame_w + graph_x)/2, centre_y + 0.18, "$Network$", fontsize=7, ha="center")
ax.annotate("", (split_x + 1.25, row_y + 0.80), (graph_x + frame_w, row_y + 0.43), arrowprops=arrow_kw)
ax.text((graph_x + frame_w + split_x + 1.2)/2, row_y + 0.95, "$Community$", fontsize=7, ha="center")

# ---------- Markov Transition ----------
mk_x, mk_y = split_x + 1.25, row_y + 0.55
markov_width, markov_height = 0.9, 0.7
rbox(ax, mk_x, mk_y, markov_width, markov_height, C_GNN_MARKOV, "Markov\nTransition")
nodes1 = [(mk_x + 0.15, mk_y + 0.15),(mk_x + 0.35, mk_y + 0.10),(mk_x + 0.25, mk_y - 0.05)]
nodes2 = [(mk_x + 0.65, mk_y + 0.20),(mk_x + 0.85, mk_y + 0.05),(mk_x + 0.75, mk_y - 0.08)]
for (cx, cy) in nodes1 + nodes2:
    ax.add_patch(Circle((cx, cy), 0.06, fc=C_NODE, ec="k", lw=0.8, zorder=3))
ax.add_patch(Ellipse((mk_x + 0.25, mk_y + 0.05), 0.35, 0.3, angle=12, fill=False, ls="--", lw=1.0, ec="#888", zorder=2))
ax.add_patch(Ellipse((mk_x + 0.75, mk_y + 0.05), 0.4, 0.35, angle=-10, fill=False, ls="--", lw=1.0, ec="#888", zorder=2))
ax.annotate("", (mk_x + 0.38, mk_y + 0.05), (mk_x + 0.62, mk_y + 0.08),
            arrowprops=dict(arrowstyle="->", lw=1.0, ls="--", color="#333"))

# ----- Heatmap directly below Markov -----
heat_w, heat_h = 0.7, 0.35
heat_x = mk_x + 0.1
heat_y = mk_y - heat_h - 0.2
heat_ax = ax.inset_axes([heat_x, heat_y, heat_w, heat_h], transform=ax.transData)
markov_matrix = np.array([[0.3, 0.8],[0.4, 0.7]])
heat_ax.imshow(markov_matrix, cmap="Greens", vmin=0, vmax=1)
heat_ax.set_xticks([0,1]); heat_ax.set_yticks([0,1])
heat_ax.set_xticklabels(["C1","C2"],fontsize=6); heat_ax.set_yticklabels(["C1","C2"],fontsize=6)
heat_ax.set_title("Markov Matrix",fontsize=7)
heat_ax.tick_params(axis='both',which='both',length=0)

# ----- Refined graph architecture below heatmap -----
net_y = heat_y - 0.55
cluster_left = [(mk_x + 0.2, net_y),
                (mk_x + 0.4, net_y + 0.15),
                (mk_x + 0.35, net_y - 0.18)]
cluster_right = [(mk_x + 0.7, net_y + 0.10),
                 (mk_x + 0.85, net_y - 0.10),
                 (mk_x + 0.95, net_y + 0.08)]
# Draw edges
for (x1,y1) in cluster_left:
    for (x2,y2) in cluster_left:
        if (x1, y1) != (x2, y2):
            ax.plot([x1, x2], [y1, y2], lw=0.8, color="#999")
for (x1,y1) in cluster_right:
    for (x2,y2) in cluster_right:
        if (x1, y1) != (x2, y2):
            ax.plot([x1, x2], [y1, y2], lw=0.8, color="#999")
# Draw nodes
for (cx, cy) in cluster_left + cluster_right:
    ax.add_patch(Circle((cx, cy), 0.06, fc=C_NODE, ec="k", lw=0.8))

# Ellipse indicating communities
ax.add_patch(Ellipse((mk_x + 0.32, net_y - 0.02), 0.35, 0.4, angle=10, fill=False, ls="--", lw=1.0, ec="#555"))
ax.add_patch(Ellipse((mk_x + 0.83, net_y), 0.4, 0.45, angle=-10, fill=False, ls="--", lw=1.0, ec="#555"))
ax.text(mk_x + 0.32, net_y - 0.32, "A1", fontsize=7, ha="center")
ax.text(mk_x + 0.83, net_y + 0.35, "A2", fontsize=7, ha="center")

# ----- Dashed frame around Markov + heatmap + network -----
frame_left = mk_x - 0.10
frame_bottom = net_y - 0.35
frame_w_total = 1.15
frame_h_total = (mk_y + markov_height) - frame_bottom
ax.add_patch(Rectangle((frame_left, frame_bottom), frame_w_total, frame_h_total,
                       fill=False, ls="--", lw=1.0))

# ---------- PPO Policy Network ----------
ax.add_patch(Rectangle((split_x + 4, centre_y -0.5), 0.35, 0.13, fc=C_BAR1, ec="k"))
ax.add_patch(Rectangle((split_x + 4.35, centre_y -0.5), 0.35, 0.13, fc=C_BAR2, ec="k"))
ax.text(split_x + 4.3, centre_y - 0.4, "${PPO}$", fontsize=15, ha="center", va="bottom")
nn_x, nn_y = split_x + 4.15, centre_y + 0.6
layer_w, layer_h = 0.5, 0.18
layers = ["Input\n($s_v$)", "Dense", "ReLU", "Output\n$\\pi(a|s_v)$"]
for i, lbl in enumerate(layers):
    ly = nn_y - i*(layer_h + 0.08)
    ax.add_patch(Rectangle((nn_x, ly), layer_w, layer_h, fc="#fbe9e7", ec="k"))
    ax.text(nn_x + layer_w/2, ly + layer_h/2, lbl, ha="center", va="center", fontsize=6)
    if i < len(layers)-1:
        ax.annotate("", (nn_x + layer_w/2, ly), (nn_x + layer_w/2, ly - 0.08), arrowprops=arrow_kw)
ax.text(nn_x + layer_w/2, nn_y + 0.18, "PPO\n$\\Policy Network", ha="center", fontsize=7, weight="bold")

# ---------- Tide‑Mark Classifier -----------
class_x = split_x + 5.6
class_w, class_h = 1.2, 0.85
class_y = centre_y - class_h/2
ax.add_patch(Rectangle((class_x, class_y), class_w, class_h, fill=False, lw=1.1))
ax.text(class_x + class_w/2, class_y + class_h + 0.05, "Tide‑Mark Classifier", fontsize=8, weight="bold", ha="center")
inner_x = class_x + 0.15
inner_w = class_w - 0.30
ly_top = class_y + class_h - 0.15
step_h, gap = 0.14, 0.05
class_layers = ["Input", "Dense", "ReLU", "Output"]
for i, lbl in enumerate(class_layers):
    ly = ly_top - i*(step_h + gap)
    ax.add_patch(Rectangle((inner_x, ly), inner_w, step_h, fc="#fbe9e7", ec="k"))
    ax.text(inner_x + inner_w/2, ly + step_h/2, lbl, ha="center", va="center", fontsize=6)
    if i < len(class_layers)-1:
        ax.annotate("", (inner_x + inner_w/2, ly), (inner_x + inner_w/2, ly - gap), arrowprops=arrow_kw)

# Arrow PPO -> Classifier
ax.annotate("", (class_x, centre_y), (split_x + 4.7, centre_y), arrowprops=arrow_kw)
ax.text((class_x + split_x + 4.7)/2, centre_y + 0.18, "$Final$", fontsize=7, ha="center")

# Arrow Markov -> PPO
ax.annotate("", (split_x + 4.0, centre_y), (mk_x + markov_width/2, centre_y), arrowprops=arrow_kw)
ax.text((mk_x + markov_width/2 + split_x + 4.0)/2, centre_y + 0.18, "$State$", fontsize=7, ha="center")

plt.tight_layout()
plt.show()
