import scanpy as sc
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import plotly.express as px

def plot_embedding(data, labels, title):
    fig = px.scatter(
        data, 
        x='x', 
        y='y', 
        color=labels.astype(str),
        title=title,
        labels={'color': 'Cell Type'}
    )
    fig.update_traces(marker=dict(size=5, line=dict(width=1, color='DarkSlateGrey')))
    fig.show()


# Setup logging
logging.basicConfig(filename='data_processing_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')

# Load data
logging.info("Loading data")
adata = sc.read_csv('data.csv')
logging.info(f"Data loaded with shape {adata.shape}")

# Quality control
logging.info("Starting quality control")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
adata = adata[adata.obs.n_genes_by_counts < 2500, :].copy()
adata = adata[adata.obs.pct_counts_mt < 5, :].copy()
logging.info("Quality control completed")

# Normalization
logging.info("Normalizing data")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
logging.info("Normalization completed")

# Feature Selection
logging.info("Starting feature selection")
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var.highly_variable].copy()
logging.info("Feature selection completed")

# Dimensionality Reduction
logging.info("Starting dimensionality reduction")
sc.tl.pca(adata, svd_solver='arpack')
sc.tl.tsne(adata, n_pcs=50)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
logging.info("Dimensionality reduction completed")

# Generate synthetic cell types using K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(adata.obsm['X_pca'])
adata.obs['cell_type'] = kmeans.labels_.astype(str)
adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')
silhouette_avg = silhouette_score(adata.obsm['X_pca'], adata.obs['cell_type'].cat.codes)
logging.info(f"Synthetic cell type labels generated using K-means clustering, silhouette score: {silhouette_avg}")

# Graph Neural Network setup
features = torch.tensor(adata.X, dtype=torch.float)
edge_index = torch.tensor(np.array(adata.obsp['connectivities'].nonzero()), dtype=torch.long)
data = Data(x=features, edge_index=edge_index)
data.y = torch.tensor(adata.obs['cell_type'].cat.codes, dtype=torch.long)

# Define GCN model
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(data.num_features, 16)
        self.conv2 = GCNConv(16, len(np.unique(kmeans.labels_)))

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Prepare for training and evaluation
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
losses = []
accuracies = []

# Split data for training and validation
train_mask = np.random.rand(len(data.y)) < 0.8
val_mask = ~train_mask

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    # Validation
    model.eval()
    with torch.no_grad():
        pred = out[val_mask].max(1)[1]
        acc = accuracy_score(data.y[val_mask].numpy(), pred.numpy())
        accuracies.append(acc)
    logging.info(f'Epoch: {epoch}, Training Loss: {loss.item()}, Validation Accuracy: {acc}')

# Plot training loss and validation accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(losses, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot PCA, t-SNE, UMAP
pca_df = pd.DataFrame(adata.obsm['X_pca'][:, :2], columns=['x', 'y'])
tsne_df = pd.DataFrame(adata.obsm['X_tsne'], columns=['x', 'y'])
umap_df = pd.DataFrame(adata.obsm['X_umap'], columns=['x', 'y'])
plot_embedding(pca_df, adata.obs['cell_type'].cat.codes, 'PCA')
plot_embedding(tsne_df, adata.obs['cell_type'].cat.codes, 't-SNE')
plot_embedding(umap_df, adata.obs['cell_type'].cat.codes, 'UMAP')

# Save final results
adata.write('processed_data.h5ad')
logging.info("All processing steps completed and data saved")
