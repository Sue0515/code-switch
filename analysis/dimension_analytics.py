import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

def run_pca_analysis(all_embeddings, labels, n_components=2):
    """
    Run PCA analysis 
    """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(all_embeddings)
    
    # Create DataFrame for easier plotting
    pca_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'Language': labels
    })
    
    return pca_df, pca

def run_tsne_analysis(all_embeddings, labels, perplexity=30, n_iter=1000, random_state=42):
    """
    Run t-SNE analysis
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
    tsne_result = tsne.fit_transform(all_embeddings)
    
    tsne_df = pd.DataFrame({
        'TSNE1': tsne_result[:, 0],
        'TSNE2': tsne_result[:, 1],
        'Language': labels
    })
    
    return tsne_df

def analyze_pca_dimensions(all_embeddings, n_components=20):
    """
    Analyze PCA dimension importance
    """
    pca_full = PCA(n_components=n_components)
    pca_full.fit(all_embeddings)
    return pca_full

def run_kmeans_clustering(all_embeddings, n_clusters=4, random_state=42, n_init=10):
    """
    Run K-means clustering on embeddings
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    clusters = kmeans.fit_predict(all_embeddings)
    return clusters

def purity_score(true_labels, pred_labels):
    """
    Calculate cluster purity score
    """
    # Convert labels to numpy arrays
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    # Create label to cluster mapping
    label_to_cluster = {}
    for i, label in enumerate(set(true_labels)):
        # Find most common cluster for this label
        indices = np.where(true_labels == label)[0]
        clusters_for_label = pred_labels[indices]
        most_common_cluster = np.bincount(clusters_for_label).argmax()
        label_to_cluster[label] = most_common_cluster
    
    # Map true labels to their preferred cluster
    mapped_labels = np.array([label_to_cluster[label] for label in true_labels])
    
    # Calculate accuracy
    return np.mean(mapped_labels == pred_labels)

def run_umap_analysis(all_embeddings, labels, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Run UMAP dimensionality reduction (if umap-learn is installed)
    """
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
        umap_result = reducer.fit_transform(all_embeddings)
        
        umap_df = pd.DataFrame({
            'UMAP1': umap_result[:, 0],
            'UMAP2': umap_result[:, 1],
            'Language': labels
        })
        
        return umap_df
    except ImportError:
        print("UMAP is not installed. Skipping UMAP analysis.")
        return None

def run_hierarchical_clustering(all_embeddings, n_clusters=4):
    """
    Run hierarchical clustering on embeddings
    """
    from sklearn.cluster import AgglomerativeClustering
    
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = clustering.fit_predict(all_embeddings)
    return clusters