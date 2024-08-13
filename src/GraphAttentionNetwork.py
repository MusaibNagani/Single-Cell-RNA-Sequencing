import scanpy as sc
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import logging

# Setup logging
logging.basicConfig(filename='data_processing_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')

class GATModel(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(num_features, 8, heads=8, concat=True)
        self.gat2 = GATConv(8 * 8, num_classes, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)

def plot_embedding(data, labels, title, dimensions='2d'):
    if dimensions == '3d':
        fig = px.scatter_3d(data, x='x', y='y', z='z', color=labels.astype(str), title=title)
    else:
        fig = px.scatter(data, x='x', y='y', color=labels.astype(str), title=title)
    fig.update_traces(marker=dict(size=5, line=dict(width=1, color='DarkSlateGrey')))
    fig.show()

# Load and preprocess data
adata = sc.read_csv('data.csv')
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Feature selection and dimensionality reduction
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var.highly_variable].copy()
sc.tl.pca(adata, svd_solver='arpack', n_comps=50)
sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_pca')

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(adata.obsm['X_pca'])
adata.obs['cell_type'] = pd.Categorical(kmeans.labels_)
silhouette_avg = silhouette_score(adata.obsm['X_pca'], adata.obs['cell_type'].cat.codes)
logging.info(f"Clustering completed with silhouette score: {silhouette_avg}")

# Setup graph data
features = torch.tensor(adata.X, dtype=torch.float)
edge_index = torch.tensor(np.array(adata.obsp['connectivities'].nonzero()), dtype=torch.long)
data = Data(x=features, edge_index=edge_index)
data.y = torch.tensor(adata.obs['cell_type'].cat.codes, dtype=torch.long)

# Initialize model
model = GATModel(features.shape[1], len(np.unique(data.y.numpy())))
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Train model
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()
    logging.info(f'Epoch {epoch}: Loss {loss.item()}')

# Visualization of embeddings
model.eval()
with torch.no_grad():
    embeddings = model(data.x, data.edge_index).detach().numpy()

# Apply PCA, t-SNE, and UMAP on the embeddings for visualization
embeddings_adata = sc.AnnData(embeddings)
if embeddings.shape[0] > 2:
    sc.pp.neighbors(embeddings_adata, n_neighbors=min(15, embeddings.shape[0] - 1))
    sc.tl.pca(embeddings_adata, n_comps=min(min(embeddings.shape[0], embeddings.shape[1], 3), 2))
    sc.tl.tsne(embeddings_adata, n_pcs=min(min(embeddings.shape[0], embeddings.shape[1], 3), 2))
    sc.tl.umap(embeddings_adata, n_components=min(min(embeddings.shape[0], embeddings.shape[1], 3), 2))

    plot_embedding(pd.DataFrame(embeddings_adata.obsm['X_pca'], columns=['x', 'y']), adata.obs['cell_type'].cat.codes, 'GAT Embedded 2D PCA')
    plot_embedding(pd.DataFrame(embeddings_adata.obsm['X_tsne'], columns=['x', 'y']), adata.obs['cell_type'].cat.codes, 'GAT Embedded 2D t-SNE')
    plot_embedding(pd.DataFrame(embeddings_adata.obsm['X_umap'], columns=['x', 'y']), adata.obs['cell_type'].cat.codes, 'GAT Embedded 2D UMAP')
else:
    logging.warning("Not enough samples to compute neighbors and perform UMAP.")

# Save results
adata.write('processed_data_with_gat.h5ad')
logging.info("All processing steps completed and data saved")
