import scanpy as sc
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import logging

# Setup logging
logging.basicConfig(filename='data_processing_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')

class GCNGAT(torch.nn.Module):
    def __init__(self):
        super(GCNGAT, self).__init__()
        self.conv1 = GCNConv(data.num_features, 16)
        self.conv2 = GATConv(16, 16, heads=4, concat=True)
        self.out = torch.nn.Linear(16 * 4, 3)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(self.out(x), dim=1)

def plot_embedding_2d_3d(adata, method, title):
    if '3d' in title.lower():
        df = pd.DataFrame(adata.obsm[method][:, :3], columns=['x', 'y', 'z'])
        fig = px.scatter_3d(df, x='x', y='y', z='z', color=adata.obs['cell_type'].astype(str), title=f'{title}')
    else:
        df = pd.DataFrame(adata.obsm[method][:, :2], columns=['x', 'y'])
        fig = px.scatter(df, x='x', y='y', color=adata.obs['cell_type'].astype(str), title=f'{title}')
    fig.update_traces(marker=dict(size=5, line=dict(width=1, color='DarkSlateGrey')))
    fig.show()

# Load data
adata = sc.read_csv('data.csv')

# Preprocess and quality control
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
adata = adata[adata.obs.n_genes_by_counts < 2500, :].copy()
adata = adata[adata.obs.pct_counts_mt < 5, :].copy()

# Normalization
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Feature selection
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var.highly_variable].copy()

# Dimensionality reduction
sc.tl.pca(adata, svd_solver='arpack', n_comps=3)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=3)
sc.tl.umap(adata, n_components=3)
sc.tl.tsne(adata, n_pcs=3)

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(adata.obsm['X_pca'])
adata.obs['cell_type'] = pd.Categorical(kmeans.labels_)
silhouette_avg = silhouette_score(adata.obsm['X_pca'], adata.obs['cell_type'].cat.codes)
logging.info(f"Clustering Silhouette Score: {silhouette_avg}")

# Graph Neural Network setup
features = torch.tensor(adata.X, dtype=torch.float)
edge_index = torch.tensor(np.array(adata.obsp['connectivities'].nonzero()), dtype=torch.long)
data = Data(x=features, edge_index=edge_index)
data.y = torch.tensor(adata.obs['cell_type'].cat.codes, dtype=torch.long)

# Define masks for training and validation
num_nodes = data.y.shape[0]
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
num_train = int(num_nodes * 0.8)
train_indices = torch.randperm(num_nodes)[:num_train]
val_indices = torch.randperm(num_nodes)[num_train:]
train_mask[train_indices] = True
val_mask[val_indices] = True
data.train_mask = train_mask
data.val_mask = val_mask

# Training setup
model = GCNGAT()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = out.argmax(dim=1)
        valid_acc = accuracy_score(data.y[data.val_mask].numpy(), predictions[data.val_mask].numpy())
        logging.info(f'Epoch {epoch}: Validation Accuracy: {valid_acc}')

# Visualization of results
plot_embedding_2d_3d(adata, 'X_pca', 'PCA 3D')
plot_embedding_2d_3d(adata, 'X_pca', 'PCA 2D')
plot_embedding_2d_3d(adata, 'X_tsne', 't-SNE 2D')
plot_embedding_2d_3d(adata, 'X_umap', 'UMAP 3D')
plot_embedding_2d_3d(adata, 'X_umap', 'UMAP 2D')
