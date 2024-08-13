import scanpy as sc
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score, classification_report
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import logging

# Setup logging
def setup_logging():
    logging.basicConfig(filename='data_processing_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class GCNGAT(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCNGAT, self).__init__()
        self.gcn1 = GCNConv(num_features, 16)
        self.gcn2 = GCNConv(16, 16)
        self.gat1 = GATConv(num_features, 8, heads=8, concat=True)
        self.gat2 = GATConv(64, 16, heads=1, concat=True)
        self.fc = torch.nn.Linear(32, num_classes)

    def forward(self, x, edge_index):
        gcn_x = F.relu(self.gcn1(x, edge_index))
        gcn_x = F.dropout(gcn_x, training=self.training)
        gcn_x = self.gcn2(gcn_x, edge_index)
        
        gat_x = F.elu(self.gat1(x, edge_index))
        gat_x = F.dropout(gat_x, training=self.training)
        gat_x = self.gat2(gat_x, edge_index)
        
        combined_x = torch.cat([gcn_x, gat_x], dim=1)
        return F.log_softmax(self.fc(combined_x), dim=1)

def preprocess_data(filepath):
    adata = sc.read_csv(filepath)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.n_genes_by_counts < 2500, :].copy()
    adata = adata[adata.obs.pct_counts_mt < 5, :].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    return adata[:, adata.var.highly_variable].copy()

def perform_clustering(adata):
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_pca')
    sc.tl.tsne(adata, n_pcs=50)
    sc.tl.umap(adata)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(adata.obsm['X_pca'])
    adata.obs['cell_type'] = pd.Categorical(kmeans.labels_)
    silhouette = silhouette_score(adata.obsm['X_pca'], adata.obs['cell_type'].cat.codes)
    ari = adjusted_rand_score(adata.obs['cell_type'].cat.codes, kmeans.labels_)
    nmi = normalized_mutual_info_score(adata.obs['cell_type'].cat.codes, kmeans.labels_)
    logging.info(f"Clustering completed with silhouette score: {silhouette}, ARI: {ari}, NMI: {nmi}")
    return adata

def train_model(features, edge_index, labels, num_features, num_classes):
    model = GCNGAT(num_features, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    early_stopper = EarlyStopping(patience=10, verbose=True)
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(features, edge_index)
        loss = F.nll_loss(out, labels)
        loss.backward()
        optimizer.step()
        logging.info(f'Epoch {epoch+1}, Loss: {loss.item()}')
        if early_stopper(loss, model):
            logging.info('Early stopping')
            break
    return model

def plot_embedding(data, title):
    fig = px.scatter(data, x='x', y='y', color=data['labels'].astype(str), title=title, labels={'color': 'Cell Type'})
    fig.update_traces(marker=dict(size=5, line=dict(width=1, color='DarkSlateGrey')))
    fig.show()

def main():
    setup_logging()
    adata = sc.read_csv('data.csv')
    adata = preprocess_data('data.csv')
    adata = perform_clustering(adata)

    features = torch.tensor(adata.X, dtype=torch.float)
    edge_index = torch.tensor(np.array(adata.obsp['connectivities'].nonzero()), dtype=torch.long)
    labels = torch.tensor(adata.obs['cell_type'].cat.codes, dtype=torch.long)
    
    model = train_model(features, edge_index, labels, features.shape[1], len(np.unique(labels.numpy())))
    torch.save(model.state_dict(), 'gcn_gat_model.pth')
    
    # Evaluate model performance
    model.eval()
    with torch.no_grad():
        embeddings = model(features, edge_index).detach().numpy()
    
    pca_result = PCA(n_components=2).fit_transform(embeddings)
    tsne_result = TSNE(n_components=2).fit_transform(embeddings)
    umap_result = UMAP(n_components=2).fit_transform(embeddings)

    pca_df = pd.DataFrame(pca_result, columns=['x', 'y'])
    pca_df['labels'] = labels.numpy().astype(str)
    tsne_df = pd.DataFrame(tsne_result, columns=['x', 'y'])
    tsne_df['labels'] = labels.numpy().astype(str)
    umap_df = pd.DataFrame(umap_result, columns=['x', 'y'])
    umap_df['labels'] = labels.numpy().astype(str)

    plot_embedding(pca_df, 'PCA Results')
    plot_embedding(tsne_df, 't-SNE Results')
    plot_embedding(umap_df, 'UMAP Results')
    adata.write('processed_data.h5ad')
    logging.info("All processing steps completed and data saved")

if __name__ == "__main__":
    main()

