import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_content_vs_language_effect(within_content_similarities, across_content_similarities, 
                                  filename='content_vs_language_effect.png'):
    """
    Plot content vs. language effect on embedding similarity
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(within_content_similarities, color='blue', label='Within Same Content', 
                 alpha=0.5, kde=True, bins=30)
    sns.histplot(across_content_similarities, color='red', label='Across Different Content', 
                 alpha=0.5, kde=True, bins=30)
    plt.title('Content vs. Language Effect on Embedding Similarity', fontsize=15)
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('results', filename), dpi=300)
    plt.close()

def plot_code_switching_bias(code_switch_df, filename='code_switching_bias.png'):
    """
    Plot code-switching bias distribution
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(code_switch_df['KtoE_Bias'], color='blue', label='KtoE Bias', alpha=0.5, kde=True, bins=20)
    sns.histplot(code_switch_df['EtoK_Bias'], color='red', label='EtoK Bias', alpha=0.5, kde=True, bins=20)
    plt.axvline(0, color='black', linestyle='--')
    plt.title('Code-Switching Language Bias Distribution', fontsize=15)
    plt.xlabel('Bias (Positive = Closer to Korean, Negative = Closer to English)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('results', filename), dpi=300)
    plt.close()

def plot_code_switch_correlation(code_switch_df, filename='code_switch_correlation.png'):
    """
    Plot the correlation between different code-switching similarities
    """
    # Create scatterplot matrix
    similarity_cols = [
        'KtoE_Korean_Similarity', 'KtoE_English_Similarity', 
        'EtoK_Korean_Similarity', 'EtoK_English_Similarity'
    ]
    
    plt.figure(figsize=(14, 14))
    sns.pairplot(
        code_switch_df[similarity_cols],
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
        diag_kws={'fill': True, 'alpha': 0.6}
    )
    plt.suptitle('Relationships Between Code-Switching Similarities', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join('results', filename), dpi=300)
    plt.close()

def plot_top_biased_sentences(code_switch_df, top_n=5, filename_prefix='code_switch_top_biased'):
    """
    Plot sentences with the strongest bias in each direction
    """
    # Function to plot top sentences for a specific bias
    def plot_top_for_bias(bias_col, title, filename):
        # Get most Korean-biased and most English-biased
        most_korean = code_switch_df.nlargest(top_n, bias_col)
        most_english = code_switch_df.nsmallest(top_n, bias_col)
        
        # Combine with a label
        most_korean['Bias Direction'] = 'Korean-biased'
        most_english['Bias Direction'] = 'English-biased'
        plot_data = pd.concat([most_korean, most_english])
        
        # Create plot
        plt.figure(figsize=(14, 8))
        sns.barplot(
            data=plot_data,
            x='English_Text',
            y=bias_col,
            hue='Bias Direction',
            palette=['blue', 'red']
        )
        plt.title(title, fontsize=15)
        plt.xlabel('Content', fontsize=12)
        plt.ylabel('Bias Value', fontsize=12)
        plt.axhline(0, color='black', linestyle='--')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join('results', filename), dpi=300)
        plt.close()
    
    # Plot for KtoE bias
    plot_top_for_bias(
        'KtoE_Bias',
        f'Top {top_n} Sentences with Strongest KtoE Bias',
        f'{filename_prefix}_ktoe.png'
    )
    
    # Plot for EtoK bias
    plot_top_for_bias(
        'EtoK_Bias',
        f'Top {top_n} Sentences with Strongest EtoK Bias',
        f'{filename_prefix}_etok.png'
    )

def plot_bias_vs_similarity(code_switch_df, filename='bias_vs_similarity.png'):
    """
    Plot the relationship between bias and average similarity
    """
    # Calculate average similarity for each content
    code_switch_df['Average_Similarity'] = (
        code_switch_df['KtoE_Korean_Similarity'] + 
        code_switch_df['KtoE_English_Similarity'] +
        code_switch_df['EtoK_Korean_Similarity'] + 
        code_switch_df['EtoK_English_Similarity']
    ) / 4
    
    # Calculate absolute bias (magnitude)
    code_switch_df['KtoE_Abs_Bias'] = np.abs(code_switch_df['KtoE_Bias'])
    code_switch_df['EtoK_Abs_Bias'] = np.abs(code_switch_df['EtoK_Bias'])
    
    # Create plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=code_switch_df,
        x='Average_Similarity',
        y='KtoE_Abs_Bias',
        color='blue',
        alpha=0.7,
        label='KtoE Bias Magnitude'
    )
    sns.scatterplot(
        data=code_switch_df,
        x='Average_Similarity',
        y='EtoK_Abs_Bias',
        color='red',
        alpha=0.7,
        label='EtoK Bias Magnitude'
    )
    
    # Add trend lines
    sns.regplot(
        data=code_switch_df,
        x='Average_Similarity',
        y='KtoE_Abs_Bias',
        scatter=False,
        color='blue'
    )
    sns.regplot(
        data=code_switch_df,
        x='Average_Similarity',
        y='EtoK_Abs_Bias',
        scatter=False,
        color='red'
    )
    
    plt.title('Relationship Between Semantic Similarity and Code-Switching Bias', fontsize=15)
    plt.xlabel('Average Semantic Similarity', fontsize=12)
    plt.ylabel('Bias Magnitude (Absolute Value)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('results', filename), dpi=300)
    plt.close()

def plot_language_dominance(code_switch_df, filename='language_dominance.png'):
    """
    Visualize which language dominates in code-switched text
    """
    # Create summary data
    summary = {
        'KtoE': {
            'Korean_Dominant': (code_switch_df['KtoE_Bias'] > 0).sum(),
            'English_Dominant': (code_switch_df['KtoE_Bias'] < 0).sum(),
            'Neutral': (code_switch_df['KtoE_Bias'] == 0).sum()
        },
        'EtoK': {
            'Korean_Dominant': (code_switch_df['EtoK_Bias'] > 0).sum(),
            'English_Dominant': (code_switch_df['EtoK_Bias'] < 0).sum(),
            'Neutral': (code_switch_df['EtoK_Bias'] == 0).sum()
        }
    }
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(summary).reset_index()
    summary_df.columns = ['Dominance', 'KtoE', 'EtoK']
    
    # Melt for plotting
    plot_data = pd.melt(
        summary_df, 
        id_vars=['Dominance'],
        value_vars=['KtoE', 'EtoK'],
        var_name='Code_Switch_Type',
        value_name='Count'
    )
    
    # Create plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=plot_data,
        x='Code_Switch_Type',
        y='Count',
        hue='Dominance',
        palette={'Korean_Dominant': 'blue', 'English_Dominant': 'red', 'Neutral': 'gray'}
    )
    
    plt.title('Language Dominance in Code-Switched Text', fontsize=15)
    plt.xlabel('Code-Switching Type', fontsize=12)
    plt.ylabel('Number of Sentences', fontsize=12)
    plt.legend(title='Dominant Language')
    
    # Add count labels
    for i, p in enumerate(plt.gca().patches):
        plt.gca().annotate(
            f'{int(p.get_height())}', 
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center', fontsize=10, color='black',
            xytext=(0, 5), textcoords='offset points'
        )
        
    plt.tight_layout()
    plt.savefig(os.path.join('results', filename), dpi=300)
    plt.close()