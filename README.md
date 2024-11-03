# Unsupervised Deep Embedding for Clustering Analysis (DEC)

This repository implements **Unsupervised Deep Embedding for Clustering Analysis (DEC)**, utilizing the **Reuters10k, MNIST, and FashionMNIST** datasets. The DEC model is designed for clustering tasks by learning data representations suitable for clustering while adapting cluster assignments based on learned embeddings.

## Project Overview

Clustering, a form of unsupervised learning, groups similar data points without predefined labels. DEC integrates **deep learning with clustering** by using autoencoders to reduce dimensionality, followed by a clustering layer that iteratively refines clusters through representation learning.

### Key Features:
- **Deep embedding**: Leverages autoencoders to transform input data to lower-dimensional embeddings for effective clustering.
- **Unsupervised training**: DEC operates without labels, making it ideal for datasets without annotations.
- **Adaptive clustering**: The clustering layer fine-tunes the embeddings based on clustering assignments.

## Datasets

This project uses three distinct datasets:
- **Reuters10k**: A subset of the Reuters-21578 dataset, commonly used for text clustering tasks.
- **MNIST**: A popular dataset of handwritten digits, ideal for benchmarking clustering and classification models.
- **FashionMNIST**: An image dataset of fashion items, serving as a modern alternative to MNIST.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Kt7mmm/Unsupervised-Deep-Embedding-for-Clustering-Analysis-DEC-with-Reuters10k-MNIST-FashionMNIST-Datasets-
   cd Unsupervised-Deep-Embedding-for-Clustering-Analysis-DEC
2. Run the experiment with REUTERS10K
    python train_DEC.py --dataset reuters10k
