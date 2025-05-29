pip install torch torch_geometric torch_geometric-temporal networkx
pip install python-louvain  # Louvain
pip install gym  # PPO 示例用最简单的 Gym API

# tide_mark.py
import math, time
import networkx as nx
import numpy as np
from typing import List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric_temporal.nn.recurrent import TGNMemory, TGN
from community import community_louvain  # python-louvain

# ---------------- Stage-1  Pre-processing ---------------- #
class Preprocessor:
    def __init__(self, window_seconds: int = 3600):
        self.window = window_seconds

    def stream_to_snapshots(self, edges, node_features):
        """
        edges: List[(src, dst, ts)]  -- global unix timestamps
        node_features: Dict[node_id] -> np.array(dim)
        """
        edges.sort(key=lambda x: x[2])
        t0 = edges[0][2]
        buckets: Dict[int, List[tuple]] = {}
        for u, v, ts in edges:
            idx = (ts - t0) // self.window
            buckets.setdefault(idx, []).append((u, v))

        snapshots = []
        for t in sorted(buckets):
            e = np.array(buckets[t])
            x = np.stack([node_features[n] for n in sorted(node_features)], axis=0)
            snapshots.append((e, x))
        return snapshots

# ---------------- Stage-2  Temporal embedding (TGN) ----- #
class TGNEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=128, memory_dim=128, msg_dim=128):
        super().__init__()
        self.memory = TGNMemory(num_nodes=1,  # will reset later
                                raw_message_dimension=in_dim,
                                memory_dimension=memory_dim)
        self.tgn = TGN(memory=self.memory,
                       message_module=None,
                       aggregator_module=None,
                       raw_message_dimension=in_dim,
                       memory_dimension=memory_dim,
                       time_dimension=16,
                       embedding_dimension=out_dim)

    def forward_snap(self, edge_index, edge_ts, x):
        num_nodes = x.shape[0]
        self.memory.__init_memory__(num_nodes)  # reset
        z = self.tgn(x=x,
                     edge_index=edge_index,
                     edge_weight=None,
                     edge_timestamp=edge_ts)
        return z.detach()

# ---------------- Stage-3  Initial clustering ------------- #
def knn_graph(z: torch.Tensor, k: int = 20):
    z = z.cpu().numpy()
    n = z.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        dist = np.linalg.norm(z[i] - z, axis=1)
        nn_idx = dist.argsort()[1:k+1]
        for j in nn_idx:
            g.add_edge(i, j, weight=float(math.exp(-dist[j])))
    return g

def louvain_clusters(g: nx.Graph):
    part = community_louvain.best_partition(g, weight='weight')
    labels = np.array([part[i] for i in range(len(part))])
    return labels

# ---------------- Stage-4  Markov transition ------------- #
def transition_matrix(prev_labels, curr_labels, laplace=0.1):
    k_prev = prev_labels.max() + 1
    k_curr = curr_labels.max() + 1
    P = laplace * np.ones((k_prev, k_curr))
    for i in range(len(prev_labels)):
        P[prev_labels[i], curr_labels[i]] += 1
    P /= P.sum(axis=1, keepdims=True)
    return P  # shape [k_prev, k_curr]

# ---------------- Stage-5  PPO Boundary refinement ------- #
class BoundaryEnv:  # 简化 Gym-style 环境
    def __init__(self, g, labels, z, P):
        self.g, self.labels, self.z, self.P = g, labels, z, P
        self.boundary_nodes = [n for n in g.nodes
                               if any(labels[n] != labels[m] for m in g.neighbors(n))]
        self.ptr = 0

    def reset(self):
        self.ptr = 0
        return self._state(self.boundary_nodes[self.ptr])

    def _state(self, v):
        lbl = self.labels[v]
        return torch.cat([self.z[v], torch.tensor([lbl])])  # toy state

    def step(self, action):
        v = self.boundary_nodes[self.ptr]
        old_lbl = self.labels[v]
        new_lbl = action
        reward = 0.0
        if new_lbl != old_lbl:
            self.labels[v] = new_lbl
            reward = 1.0  # 这里应当用 ΔQ + αΔCond + βP_{old,new}
        self.ptr += 1
        done = self.ptr >= len(self.boundary_nodes)
        next_state = self._state(self.boundary_nodes[self.ptr]) if not done else None
        return next_state, reward, done, {}

class PPOAgent(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(),
                                nn.Linear(64, n_actions))
        self.v  = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(),
                                nn.Linear(64, 1))

    def forward(self, s):
        logits = self.pi(s)
        value  = self.v(s)
        return logits, value

# (完整 PPO 训练代码略，可直接用 stable-baselines3 / RLlib)

# ---------------- Orchestrator --------------------------- #
class TideMark:
    def __init__(self, feature_dim, k=20):
        self.encoder = TGNEncoder(feature_dim)
        self.k = k
        self.prev_labels = None

    def process_snapshot(self, edge_index, edge_ts, x):
        # 2) embedding
        z = self.encoder.forward_snap(edge_index, edge_ts, x)
        # 3) clustering
        g_knn = knn_graph(z, self.k)
        labels = louvain_clusters(g_knn)
        # 4) Markov
        P = transition_matrix(self.prev_labels, labels) if self.prev_labels is not None else None
        # 5) PPO refinement (demo: skipped / or plug real PPO here)
        # ---> after PPO call we’d have labels_refined
        self.prev_labels = labels
        return labels, z

# ---------------- Quick smoke-test ----------------------- #
if __name__ == "__main__":
    # 生成一个假数据流
    num_nodes = 100
    feats = {i: np.random.randn(32) for i in range(num_nodes)}
    edges = [(np.random.randint(num_nodes),
              np.random.randint(num_nodes),
              1_600_000_000 + np.random.randint(0, 5_000)) for _ in range(500)]
    pre = Preprocessor()
    snaps = pre.stream_to_snapshots(edges, feats)

    tide = TideMark(feature_dim=32)
    for t, (e, x) in enumerate(snaps):
        ei = torch.tensor(e.T, dtype=torch.long)  # shape [2, E]
        ts = torch.tensor(np.random.rand(e.shape[0]), dtype=torch.float)
        x  = torch.tensor(x, dtype=torch.float)
        lbl, z = tide.process_snapshot(ei, ts, x)
        print(f"[t={t}] communities =", lbl.max()+1)
