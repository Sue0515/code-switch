import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_similarity_distribution(similarity_df, filename='cosine_similarity_distribution.png'):
    """
    Plot distribution of cosine similarities
    """
    plt.figure(figsize=(15, 10))
    melted_similarities = similarity_df.drop('Pair_ID', axis=1).melt(
        var_name='Pair Type', 
        value_name='Cosine Similarity'
    )
    
    sns.boxplot(x='Pair Type', y='Cosine Similarity', data=melted_similarities)
    plt.title('Distribution of Cosine Similarities Between Different Language Pairs', fontsize=15)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join('results', filename), dpi=300)
    plt.close()

def plot_distance_distribution(distance_df, filename='euclidean_distance_distribution.png'):
    """
    Plot distribution of Euclidean distances
    """
    plt.figure(figsize=(15, 10))
    melted_distances = distance_df.drop('Pair_ID', axis=1).melt(
        var_name='Pair Type', 
        value_name='Euclidean Distance'
    )
    
    sns.boxplot(x='Pair Type', y='Euclidean Distance', data=melted_distances)
    plt.title('Distribution of Euclidean Distances Between Different Language Pairs', fontsize=15)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join('results', filename), dpi=300)
    plt.close()

def plot_content_similarity_distribution(content_sim_df, filename='content_similarity_distribution.png'):
    """
    Plot distribution of content similarities
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(content_sim_df['Average_Similarity'], bins=20, kde=True)
    plt.title('Distribution of Average Semantic Similarity Across Language Versions', fontsize=15)
    plt.xlabel('Average Similarity', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.axvline(content_sim_df['Average_Similarity'].mean(), color='r', linestyle='--', 
                label=f'Mean: {content_sim_df["Average_Similarity"].mean():.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('results', filename), dpi=300)
    plt.close()

def plot_similarity_heatmap(content_sim_df, filename='similarity_heatmap.png'):
    """
    Create a heatmap of similarities between language pairs
    """
    # Extract only the similarity columns
    similarity_cols = ['KO-EN', 'KO-KTOE', 'KO-ETOK', 'EN-KTOE', 'EN-ETOK', 'KTOE-ETOK']
    similarity_data = content_sim_df[similarity_cols].mean().reset_index()
    similarity_data.columns = ['Pair', 'Average Similarity']
    
    # Reshape data for heatmap
    languages = ['Korean', 'English', 'KtoE', 'EtoK']
    heatmap_data = pd.DataFrame(np.zeros((len(languages), len(languages))), 
                               index=languages, columns=languages)
    
    # Fill the heatmap data
    for lang1 in languages:
        for lang2 in languages:
            if lang1 == lang2:
                heatmap_data.loc[lang1, lang2] = 1.0  # Self-similarity is 1.0
            else:
                # Find the column name based on languages
                if lang1 == 'Korean' and lang2 == 'English':
                    col = 'KO-EN'
                elif lang1 == 'Korean' and lang2 == 'KtoE':
                    col = 'KO-KTOE'
                elif lang1 == 'Korean' and lang2 == 'EtoK':
                    col = 'KO-ETOK'
                elif lang1 == 'English' and lang2 == 'KtoE':
                    col = 'EN-KTOE'
                elif lang1 == 'English' and lang2 == 'EtoK':
                    col = 'EN-ETOK'
                elif lang1 == 'KtoE' and lang2 == 'EtoK':
                    col = 'KTOE-ETOK'
                elif lang2 == 'Korean' and lang1 == 'English':
                    col = 'KO-EN'
                elif lang2 == 'Korean' and lang1 == 'KtoE':
                    col = 'KO-KTOE'
                elif lang2 == 'Korean' and lang1 == 'EtoK':
                    col = 'KO-ETOK'
                elif lang2 == 'English' and lang1 == 'KtoE':
                    col = 'EN-KTOE'
                elif lang2 == 'English' and lang1 == 'EtoK':
                    col = 'EN-ETOK'
                elif lang2 == 'KtoE' and lang1 == 'EtoK':
                    col = 'KTOE-ETOK'
                
                # Set the value
                if col in similarity_cols:
                    heatmap_data.loc[lang1, lang2] = content_sim_df[col].mean()
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', vmin=0, vmax=1, fmt='.4f')
    plt.title('Average Semantic Similarity Between Language Types', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join('results', filename), dpi=300)
    plt.close()
    
def plot_similarity_by_content(content_sim_df, top_n=10, filename_prefix='similarity_by_content'):
    """
    Plot similarities for specific contents
    """
    # Get contents with highest and lowest average similarity
    highest_sim = content_sim_df.nlargest(top_n, 'Average_Similarity')
    lowest_sim = content_sim_df.nsmallest(top_n, 'Average_Similarity')
    
    # Function to plot similarity profile for a set of contents
    def plot_similarity_profile(contents, title, filename):
        plt.figure(figsize=(15, 8))
        
        # Select similarity columns
        similarity_cols = ['KO-EN', 'KO-KTOE', 'KO-ETOK', 'EN-KTOE', 'EN-ETOK', 'KTOE-ETOK']
        
        # Create a pivoted dataset for easier plotting
        plot_data = pd.melt(
            contents[['Content_ID', 'English_Text'] + similarity_cols], 
            id_vars=['Content_ID', 'English_Text'],
            value_vars=similarity_cols,
            var_name='Language Pair',
            value_name='Similarity'
        )
        
        # Plot
        sns.barplot(
            data=plot_data,
            x='English_Text', 
            y='Similarity',
            hue='Language Pair',
            palette='viridis'
        )
        
        plt.title(title, fontsize=15)
        plt.xlabel('Content', fontsize=12)
        plt.ylabel('Cosine Similarity', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Language Pair')
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(os.path.join('results', filename), dpi=300)
        plt.close()
    
    # Plot for highest and lowest similarity contents
    plot_similarity_profile(
        highest_sim, 
        f'Top {top_n} Contents with Highest Cross-Language Similarity',
        f'{filename_prefix}_highest.png'
    )
    
    plot_similarity_profile(
        lowest_sim, 
        f'Top {top_n} Contents with Lowest Cross-Language Similarity',
        f'{filename_prefix}_lowest.png'
    )

def plot_similarity_by_pair(similarity_df, distance_df, filename='similarity_by_pair_comparison.png'):
    """
    Compare cosine similarity and Euclidean distance for each language pair
    """
    # Prepare average metrics per pair
    avg_similarities = similarity_df.drop('Pair_ID', axis=1).mean()
    avg_distances = distance_df.drop('Pair_ID', axis=1).mean()
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Pair': avg_similarities.index,
        'Cosine Similarity': avg_similarities.values,
        'Euclidean Distance': avg_distances.values
    })
    
    # Normalize Euclidean distances to 0-1 scale for better visualization
    min_dist = comparison_df['Euclidean Distance'].min()
    max_dist = comparison_df['Euclidean Distance'].max()
    comparison_df['Normalized Distance'] = (comparison_df['Euclidean Distance'] - min_dist) / (max_dist - min_dist)
    
    # Plot
    plt.figure(figsize=(14, 8))
    
    # Create a twin axes
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot each metric
    sns.barplot(
        data=comparison_df,
        x='Pair',
        y='Cosine Similarity',
        color='blue',
        alpha=0.7,
        ax=ax1,
        label='Cosine Similarity'
    )
    
    sns.barplot(
        data=comparison_df,
        x='Pair',
        y='Normalized Distance',
        color='red',
        alpha=0.7,
        ax=ax2,
        label='Euclidean Distance (Normalized)'
    )
    
    # Labels and title
    ax1.set_xlabel('Language Pair', fontsize=12)
    ax1.set_ylabel('Cosine Similarity', fontsize=12, color='blue')
    ax2.set_ylabel('Normalized Euclidean Distance', fontsize=12, color='red')
    ax1.set_title('Comparison of Similarity and Distance Metrics by Language Pair', fontsize=15)
    
    # Adjust ticks
    plt.xticks(rotation=45)
    
    # Custom legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='blue', lw=4, alpha=0.7),
        Line2D([0], [0], color='red', lw=4, alpha=0.7)
    ]
    ax1.legend(custom_lines, ['Cosine Similarity', 'Euclidean Distance (Normalized)'])
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', filename), dpi=300)
    plt.close()

def plot_similarity_correlation(content_sim_df, filename='similarity_correlation.png'):
    """
    Plot the correlation matrix of similarities between different language pairs
    """
    # Select only the similarity columns
    similarity_cols = ['KO-EN', 'KO-KTOE', 'KO-ETOK', 'EN-KTOE', 'EN-ETOK', 'KTOE-ETOK']
    
    # Calculate correlation matrix
    corr_matrix = content_sim_df[similarity_cols].corr()
    
    # Plot the correlation matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Correlation of Similarities Between Language Pairs', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join('results', filename), dpi=300)
    plt.close()