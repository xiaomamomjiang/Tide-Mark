# Tide-Mark
1. What is TIDE-MARK?
TIDE-MARK is a five-stage pipeline that detects and continuously refines user-community structures while a social-media cascade is still unfolding:

Stage	Module	Goal	Key Paper Section
1	Preprocessor	Slice the raw edge stream into fixed-length snapshots and build initial node features.	§ Materials & Methods: Preprocessing
2	TGNEncoder	Encode each snapshot with a Temporal Graph Network (TGN) to obtain node embeddings z<sub>v</sub><sup>t</sup>.	§ Stage 2
3	louvain_clusters()	Build a k-NN similarity graph on the embeddings, then run Louvain to get a coarse partition C<sub>t</sub><sup>init</sup>.	§ Stage 3
4	transition_matrix()	Estimate a Markov transition matrix P<sup>(t)</sup> that captures how communities evolve between snapshots.	§ Stage 4
5	BoundaryEnv + PPOAgent	Treat boundary-node reassignment as an RL problem; train a PPO agent that maximises ΔModularity + α ΔConductance + β P.	§ Stage 5

The skeleton code you see here is deliberately light-weight: every stage is a small, self-contained Python class / function so you can swap in fancier models (DySAT → TGN, Louvain → Leiden, PPO → IMPALA, …) without touching the rest of the pipeline.

2. Installation
# Core deps
pip install torch torch_geometric torch_geometric-temporal networkx
# Louvain clustering
pip install python-louvain
# (Optional) RL helpers
pip install gym  stable-baselines3
Tested with Python 3.10, PyTorch 2.2.

3. Data format

edges : List[(src, dst, timestamp)]   # UNIX seconds
features : Dict[node_id] -> np.ndarray(shape=(F,))
Only two things are mandatory:

Edges must be timestamped (they drive window slicing).

Each node needs some feature vector (text BERT, user metadata, or even a one-hot).

4. Quick start

python tide_mark.py          # runs a toy stream with random data
Output:

[t=0] communities = 4
[t=1] communities = 5
...
5. Directory structure
tide-mark/
├── tide_mark.py          # all five stages in one file
├── README.md             # this file
└── requirements.txt      # optional pin of exact versions
6. Customising each stage
Need	What to change
Different window length	Preprocessor(window_seconds=1800)
Replace TGN with DySAT	Drop in PyG-Temporal DySAT module inside TGNEncoder.
GPU k-NN for huge graphs	Swap knn_graph() with FAISS/pyTorch-neighbor search.
Better RL	Replace the toy BoundaryEnv + PPOAgent with stable-baselines3 PPO (state = embedding ⊕ structural stats).
White-box agent	After training, collect (state, action) and fit sklearn.tree.DecisionTreeClassifier to reach ≈90 % fidelity (cf. paper § De-Blackboxing).
