import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_pca_with_clusters(pca_df, clusters, pca, filename='pca_with_clusters.png'):
    """
    Plot PCA with cluster information
    """
    pca_df_with_clusters = pca_df.copy()
    pca_df_with_clusters['Cluster'] = clusters.astype(str)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=pca_df_with_clusters, 
        x='PC1', 
        y='PC2', 
        hue='Language', 
        style='Cluster',
        palette='viridis',
        alpha=0.7,
        s=100
    )
    plt.title('PCA of Multilingual Embeddings with K-means Clusters', fontsize=15)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.legend(title='Language Type', fontsize=12, title_fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join('results', filename), dpi=300)
    plt.close()

def plot_tsne_with_clusters(tsne_df, clusters, filename='tsne_with_clusters.png'):
    """
    Plot t-SNE with cluster information
    """
    tsne_df_with_clusters = tsne_df.copy()
    tsne_df_with_clusters['Cluster'] = clusters.astype(str)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=tsne_df_with_clusters, 
        x='TSNE1', 
        y='TSNE2', 
        hue='Language', 
        style='Cluster',
        palette='viridis',
        alpha=0.7,
        s=100
    )
    plt.title('t-SNE of Multilingual Embeddings with K-means Clusters', fontsize=15)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(title='Language Type', fontsize=12, title_fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join('results', filename), dpi=300)
    plt.close()

def plot_cluster_distribution(labels, clusters, filename='cluster_distribution.png'):
    """
    Plot the distribution of language types within each cluster
    """
    # Create DataFrame with language and cluster information
    cluster_df = pd.DataFrame({
        'Language': labels,
        'Cluster': clusters.astype(str)
    })
    
    # Count occurrences of each language in each cluster
    cluster_counts = pd.crosstab(cluster_df['Cluster'], cluster_df['Language'])
    
    # Convert to percentages
    cluster_percentages = cluster_counts.div(cluster_counts.sum(axis=1), axis=0) * 100
    
    # Plot
    plt.figure(figsize=(12, 8))
    cluster_percentages.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Distribution of Language Types Within Each Cluster', fontsize=15)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Percentage', fontsize=12)
    plt.legend(title='Language Type', fontsize=12, title_fontsize=13)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join('results', filename), dpi=300)
    plt.close()

def plot_cluster_centroids(all_embeddings, clusters, pca, labels, filename='cluster_centroids.png'):
    """
    Plot cluster centroids in PCA space
    """
    # Calculate cluster centroids
    unique_clusters = np.sort(np.unique(clusters))
    centroids = np.array([all_embeddings[clusters == c].mean(axis=0) for c in unique_clusters])
    
    # Transform centroids to PCA space
    centroid_pca = pca.transform(centroids)
    
    # Create DataFrame for original data points
    pca_df = pd.DataFrame({
        'PC1': pca.transform(all_embeddings)[:, 0],
        'PC2': pca.transform(all_embeddings)[:, 1],
        'Language': labels,
        'Cluster': clusters.astype(str)
    })
    
    # Create DataFrame for centroids
    centroid_df = pd.DataFrame({
        'PC1': centroid_pca[:, 0],
        'PC2': centroid_pca[:, 1],
        'Cluster': unique_clusters.astype(str)
    })
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Plot data points with lower alpha
    sns.scatterplot(
        data=pca_df, 
        x='PC1', 
        y='PC2', 
        hue='Language', 
        style='Cluster',
        palette='viridis',
        alpha=0.3,
        s=80
    )
    
    # Plot centroids
    for i, centroid in centroid_df.iterrows():
        plt.scatter(
            centroid['PC1'], 
            centroid['PC2'], 
            marker='X', 
            color='red', 
            s=200, 
            edgecolor='black', 
            label='Centroid' if i == 0 else ""
        )
        plt.text(
            centroid['PC1'], 
            centroid['PC2'], 
            f"Cluster {centroid['Cluster']}", 
            fontsize=12, 
            ha='center', 
            va='bottom', 
            fontweight='bold'
        )
    
    plt.title('Cluster Centroids in PCA Space', fontsize=15)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    
    # Add 'Centroid' to the legend
    handles, labels_legend = plt.gca().get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='red', 
                              markersize=10, label='Centroid', markeredgecolor='black'))
    labels_legend.append('Centroid')
    
    plt.legend(handles, labels_legend, title='Language Type', fontsize=12, title_fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join('results', filename), dpi=300)
    plt.close()

def plot_cluster_silhouette(all_embeddings, clusters, filename='cluster_silhouette.png'):
    """
    Plot silhouette scores for clusters.
    
    Args:
        all_embeddings (numpy.ndarray): Combined embeddings.
        clusters (numpy.ndarray): Cluster assignments.
        filename (str): Filename for saving the plot.
    """
    try:
        from sklearn.metrics import silhouette_samples, silhouette_score
        import matplotlib.cm as cm
        
        # Calculate silhouette scores
        silhouette_avg = silhouette_score(all_embeddings, clusters)
        sample_silhouette_values = silhouette_samples(all_embeddings, clusters)
        
        # Plot
        plt.figure(figsize=(12, 8))
        y_lower = 10
        
        unique_clusters = np.sort(np.unique(clusters))
        n_clusters = len(unique_clusters)
        
        # Create a colormap
        cmap = cm.get_cmap("viridis", n_clusters)
        
        for i, cluster in enumerate(unique_clusters):
            # Aggregate silhouette scores for samples in this cluster
            ith_cluster_values = sample_silhouette_values[clusters == cluster]
            ith_cluster_values.sort()
            
            size_cluster_i = ith_cluster_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            # Fill area with color
            plt.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_values,
                facecolor=cmap(i / n_clusters),
                edgecolor=cmap(i / n_clusters),
                alpha=0.7
            )
            
            # Label cluster names
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Cluster {cluster}')
            
            # Compute new y_lower for next plot
            y_lower = y_upper + 10
        
        # Add vertical line for average silhouette score
        plt.axvline(x=silhouette_avg, color="red", linestyle="--")
        plt.text(
            silhouette_avg + 0.02, 
            plt.ylim()[0] + 0.05 * (plt.ylim()[1] - plt.ylim()[0]), 
            f'Average silhouette score: {silhouette_avg:.3f}', 
            color="red"
        )
        
        plt.title('Silhouette Plot for Cluster Evaluation', fontsize=15)
        plt.xlabel('Silhouette Coefficient Values', fontsize=12)
        plt.ylabel('Cluster Label', fontsize=12)
        plt.yticks([])  # Clear y ticks
        plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.tight_layout()
        plt.savefig(os.path.join('results', filename), dpi=300)
        plt.close()
    except ImportError:
        print("scikit-learn silhouette functions not available. Skipping silhouette plot.")