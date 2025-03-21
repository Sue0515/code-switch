#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch

from data.data_loader import load_data, extract_sentences
from embeddings.embedding import load_model, generate_all_embeddings
from analysis.dimension_analytics import (
    run_pca_analysis, run_tsne_analysis, analyze_pca_dimensions,
    run_kmeans_clustering, purity_score, run_umap_analysis, run_hierarchical_clustering
)
from analysis.similarity import calculate_pairwise_metrics, analyze_content_similarities, analyze_content_vs_language_effect
from analysis.code_switching import analyze_code_switching
from visualization.dimension_plots import (
    plot_pca, plot_tsne, plot_pca_explained_variance, plot_umap, plot_3d_pca
)
from visualization.similarity_plots import (
    plot_similarity_distribution, plot_distance_distribution, 
    plot_content_similarity_distribution, plot_similarity_heatmap,
    plot_similarity_by_content, plot_similarity_by_pair, plot_similarity_correlation
)
from visualization.clustering_plots import (
    plot_pca_with_clusters, plot_tsne_with_clusters, 
    plot_cluster_distribution, plot_cluster_centroids, plot_cluster_silhouette
)
from visualization.analysis_plots import (
    plot_content_vs_language_effect, plot_code_switching_bias,
    plot_code_switch_correlation, plot_top_biased_sentences,
    plot_bias_vs_similarity, plot_language_dominance
)

def parse_args():
    parser = argparse.ArgumentParser(description='Multilingual Embedding Analysis')
    
    parser.add_argument('--data_path', type=str, default='code-switch.json',
                        help='Path to the code-switch.json file')
    parser.add_argument('--model_name', type=str, default='intfloat/multilingual-e5-large',
                        help='Hugging Face model name')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU if available')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for generating embeddings')
    parser.add_argument('--n_clusters', type=int, default=4,
                        help='Number of clusters for K-means')
    parser.add_argument('--use_umap', action='store_true',
                        help='Run UMAP analysis (if installed)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    print("Loading data...")
    
    # Check if file exists in data directory, if not try root directory
    data_paths = [args.data_path, os.path.join('data', args.data_path)]
    data_path = next((path for path in data_paths if os.path.exists(path)), None)
    
    if data_path is None:
        raise FileNotFoundError(f"Could not find {args.data_path} in current or data directory")
    
    data = load_data(data_path)
    sentences = extract_sentences(data)
    
    # Step 2: Load model and generate embeddings
    print(f"Loading {args.model_name} model...")
    tokenizer, model = load_model(args.model_name)
    model = model.to(device)
    
    print("Generating embeddings...")
    embeddings = generate_all_embeddings(sentences, tokenizer, model, device=device, batch_size=args.batch_size)
    print(f"Embedding dimensions: {embeddings['Korean'].shape}")
    
    # Combine all embeddings for global analyses
    all_embeddings = np.vstack([
        embeddings['Korean'],
        embeddings['English'],
        embeddings['KtoE'],
        embeddings['EtoK']
    ])
    
    labels = (
        ['Korean'] * len(sentences['Korean']) +
        ['English'] * len(sentences['English']) +
        ['KtoE'] * len(sentences['KtoE']) +
        ['EtoK'] * len(sentences['EtoK'])
    )
    
    # Step 3: Dimensionality reduction analyses
    print("Running dimensionality reduction analyses...")
    pca_df, pca = run_pca_analysis(all_embeddings, labels)
    tsne_df = run_tsne_analysis(all_embeddings, labels)
    
    if args.use_umap:
        umap_df = run_umap_analysis(all_embeddings, labels)
        if umap_df is not None:
            plot_umap(umap_df)
    
    plot_3d_pca(all_embeddings, labels)

    # Step 4: Calculate pairwise metrics
    print("Calculating pairwise metrics...")
    similarity_df, distance_df = calculate_pairwise_metrics(data, embeddings)
    avg_similarities = similarity_df.mean().drop('Pair_ID')
    avg_distances = distance_df.mean().drop('Pair_ID')
    print("\nAverage Cosine Similarities:")
    for pair, value in avg_similarities.items():
        print(f"{pair}: {value:.4f}")  
    print("\nAverage Euclidean Distances:")
    for pair, value in avg_distances.items():
        print(f"{pair}: {value:.4f}")
    
    # Step 5: Clustering analysis
    print("Running clustering analysis...")
    kmeans_clusters = run_kmeans_clustering(all_embeddings, n_clusters=args.n_clusters)
    hier_clusters = run_hierarchical_clustering(all_embeddings, n_clusters=args.n_clusters)
    language_to_num = {lang: i for i, lang in enumerate(set(labels))}
    numeric_labels = [language_to_num[lang] for lang in labels]
    
    kmeans_purity = purity_score(numeric_labels, kmeans_clusters)
    hier_purity = purity_score(numeric_labels, hier_clusters)
    
    print(f"\nCluster Purity Scores:")
    print(f"K-means: {kmeans_purity:.4f}")
    print(f"Hierarchical: {hier_purity:.4f}")
    
    # Step 6: Content similarity analysis
    print("Analyzing content similarities...")
    content_sim_df = analyze_content_similarities(data, embeddings)
    
    # Step 7: Content vs. language effect analysis
    print("Analyzing content vs. language effect...")
    within_content_similarities, across_content_similarities = analyze_content_vs_language_effect(data, embeddings)
    
    # Step 8: Code-switching analysis
    print("Analyzing code-switching effects...")
    code_switch_df = analyze_code_switching(data, embeddings)

    ktoe_avg_bias = code_switch_df['KtoE_Bias'].mean()
    etok_avg_bias = code_switch_df['EtoK_Bias'].mean()
    
    print(f"\nCode-Switching Analysis:")
    print(f"KtoE Average Bias: {ktoe_avg_bias:.4f} ({'Closer to Korean' if ktoe_avg_bias > 0 else 'Closer to English'})")
    print(f"EtoK Average Bias: {etok_avg_bias:.4f} ({'Closer to Korean' if etok_avg_bias > 0 else 'Closer to English'})")
    
    # Step 9: PCA dimension analysis
    print("Analyzing PCA dimensions...")
    pca_full = analyze_pca_dimensions(all_embeddings)
    
    # Step 10: Generate visualizations
    print("Generating visualizations...")
    # Dimension plots
    plot_pca(pca_df, pca)
    plot_tsne(tsne_df)
    plot_pca_explained_variance(pca_full)
    
    # Similarity plots
    plot_similarity_distribution(similarity_df)
    plot_distance_distribution(distance_df)
    plot_content_similarity_distribution(content_sim_df)
    plot_similarity_heatmap(content_sim_df)
    plot_similarity_by_content(content_sim_df)
    plot_similarity_by_pair(similarity_df, distance_df)
    plot_similarity_correlation(content_sim_df)
    
    # Clustering plots
    plot_pca_with_clusters(pca_df, kmeans_clusters, pca)
    plot_tsne_with_clusters(tsne_df, kmeans_clusters)
    plot_cluster_distribution(labels, kmeans_clusters)
    plot_cluster_centroids(all_embeddings, kmeans_clusters, pca, labels)
    plot_cluster_silhouette(all_embeddings, kmeans_clusters)
    
    # Analysis plots
    plot_content_vs_language_effect(within_content_similarities, across_content_similarities)
    plot_code_switching_bias(code_switch_df)
    plot_code_switch_correlation(code_switch_df)
    plot_top_biased_sentences(code_switch_df)
    plot_bias_vs_similarity(code_switch_df)
    plot_language_dominance(code_switch_df)
    
    # Identify sentences with highest and lowest cross-language similarity
    content_sim_df_sorted = content_sim_df.sort_values('Average_Similarity')
    
    print("\nSentences with LOWEST cross-language similarity:")
    for _, row in content_sim_df_sorted.head(5).iterrows():
        print(f"Content ID: {row['Content_ID']}")
        print(f"Text: {row['English_Text']}")
        print(f"Average Similarity: {row['Average_Similarity']:.4f}")
        print("---")
    
    print("\nSentences with HIGHEST cross-language similarity:")
    for _, row in content_sim_df_sorted.tail(5).iloc[::-1].iterrows():
        print(f"Content ID: {row['Content_ID']}")
        print(f"Text: {row['English_Text']}")
        print(f"Average Similarity: {row['Average_Similarity']:.4f}")
        print("---")
    
    print(f"\nAnalysis complete! All results have been saved to the '{args.results_dir}' folder.")

if __name__ == "__main__":
    main()