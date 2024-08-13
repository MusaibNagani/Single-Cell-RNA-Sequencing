# Single-Cell RNA Sequencing Analysis Using Machine Learning

## Project Overview

This project is a comprehensive exploration of single-cell RNA sequencing (scRNA-seq) data using advanced machine learning techniques. The goal is to analyze and interpret high-dimensional scRNA-seq data to uncover cellular heterogeneity and underlying biological processes. By applying dimensionality reduction techniques and graph-based models, we aim to provide insights into the complex structure of gene expression data at a single-cell level.

## Project Structure

- **data/**: Contains the raw and processed scRNA-seq datasets used for analysis.
- **scripts/**: Python scripts for data preprocessing, implementing machine learning models, including dimensionality reduction and graph-based analysis.
- **models/**: Pre-trained models and scripts for training Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT).
- **results/**: Contains visualizations, model performance metrics, and other outputs from the analysis.
- **README.md**: Overview of the project, structure, and instructions for running the analysis.

## Data Description

The scRNA-seq data was obtained from the National Center for Biotechnology Information’s Gene Expression Omnibus (GEO) with accession number GSE86469. The dataset includes gene expression profiles across various experimental conditions, providing a rich resource for understanding cellular diversity.

## Methodology

1. **Data Preprocessing**:
   - Normalization, quality control, and feature selection were performed using Python scripts to prepare the data for analysis.

2. **Dimensionality Reduction**:
   - Techniques like PCA, t-SNE, and UMAP were applied via Python scripts to reduce the high-dimensional data and visualize the structure of the data.

3. **Graph-Based Models**:
   - Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) were implemented in Python to model the data as graphs, capturing complex relationships between cells.

4. **Visualization and Interpretation**:
   - The results were visualized using Python libraries such as Plotly, providing insights into the clustering and differentiation of cell types.

## Results

The project successfully demonstrated the application of advanced machine learning techniques to scRNA-seq data. The graph-based models, combined with dimensionality reduction, revealed distinct clusters corresponding to different cell types, providing a deeper understanding of cellular heterogeneity.

## Requirements

- Python 3.x
- Numpy
- Pandas
- Scikit-learn
- Torch
- PyTorch Geometric
- Plotly
- Scanpy

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/MusaibNagani/Single-Cell-RNA-Sequencing.git
   cd scRNA-seq-analysis
