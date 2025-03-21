import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_pca(pca_df, pca, filename='pca_multilingual_embeddings.png'):
    """
    Plot PCA results
    """
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=pca_df, 
        x='PC1', 
        y='PC2', 
        hue='Language', 
        palette='viridis',
        alpha=0.7,
        s=100
    )
    plt.title('PCA of Multilingual Embeddings', fontsize=15)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.legend(title='Language Type', fontsize=12, title_fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join('results', filename), dpi=300)
    plt.close()

def plot_tsne(tsne_df, filename='tsne_multilingual_embeddings.png'):
    """
    Plot t-SNE results
    """
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=tsne_df, 
        x='TSNE1', 
        y='TSNE2', 
        hue='Language', 
        palette='viridis',
        alpha=0.7,
        s=100
    )
    plt.title('t-SNE of Multilingual Embeddings', fontsize=15)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(title='Language Type', fontsize=12, title_fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join('results', filename), dpi=300)
    plt.close()

def plot_umap(umap_df, filename='umap_multilingual_embeddings.png'):
    """
    Plot UMAP results
    """
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=umap_df, 
        x='UMAP1', 
        y='UMAP2', 
        hue='Language', 
        palette='viridis',
        alpha=0.7,
        s=100
    )
    plt.title('UMAP of Multilingual Embeddings', fontsize=15)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.legend(title='Language Type', fontsize=12, title_fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join('results', filename), dpi=300)
    plt.close()

def plot_pca_explained_variance(pca_full, filename_prefix='pca_explained_variance'):
    """
    Plot PCA explained variance and cumulative explained variance
    """
    # Plot explained variance ratio
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, len(pca_full.explained_variance_ratio_) + 1), pca_full.explained_variance_ratio_)
    plt.xlabel('PCA Component', fontsize=12)
    plt.ylabel('Explained Variance Ratio', fontsize=12)
    plt.title('Explained Variance Ratio by PCA Component', fontsize=15)
    plt.xticks(range(1, len(pca_full.explained_variance_ratio_) + 1))
    plt.tight_layout()
    plt.savefig(os.path.join('results', f'{filename_prefix}.png'), dpi=300)
    plt.close()
    
    # Plot cumulative explained variance
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1), 
             np.cumsum(pca_full.explained_variance_ratio_), marker='o')
    plt.xlabel('Number of Components', fontsize=12)
    plt.ylabel('Cumulative Explained Variance', fontsize=12)
    plt.title('Cumulative Explained Variance vs. Number of PCA Components', fontsize=15)
    plt.xticks(range(1, len(pca_full.explained_variance_ratio_) + 1))
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Variance Threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('results', f'{filename_prefix}_cumulative.png'), dpi=300)
    plt.close()

def plot_3d_pca(all_embeddings, labels, filename='pca_3d_multilingual_embeddings.png'):
    """
    Create a 3D PCA plot
    """
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D
    
    # Run PCA with 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(all_embeddings)
    
    # Create a mapping from language to color
    languages = list(set(labels))
    colormap = plt.cm.viridis
    colors = [colormap(i/len(languages)) for i in range(len(languages))]
    language_to_color = {lang: colors[i] for i, lang in enumerate(languages)}
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each language type
    for lang in languages:
        indices = [i for i, l in enumerate(labels) if l == lang]
        ax.scatter(
            pca_result[indices, 0],
            pca_result[indices, 1],
            pca_result[indices, 2],
            label=lang,
            color=language_to_color[lang],
            alpha=0.7,
            s=50
        )
    
    # Set labels and title
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)', fontsize=12)
    ax.set_title('3D PCA of Multilingual Embeddings', fontsize=15)
    
    # Add legend
    plt.legend(title='Language Type', fontsize=12, title_fontsize=13)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join('results', filename), dpi=300)
    plt.close()