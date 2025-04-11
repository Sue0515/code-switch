import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging

logger = logging.getLogger(__name__)

class EmbeddingVisualizer:

    def __init__(self, output_dir):
       
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_loss_history(self, loss_history):
       
        plt.figure(figsize=(12, 6))
        
        # Plot total loss
        plt.plot(loss_history["epoch"], loss_history["total_loss"], 
                'o-', linewidth=2, label='Total Loss', color='navy')
        
        # Define colors for different loss components
        colors = {
            "contrastive_loss": "forestgreen",
            "cs_reg_loss": "darkorange"
        }
        
        # Plot loss components
        for component in ["contrastive_loss", "cs_reg_loss"]:
            if component in loss_history and any(loss_history[component]):
                plt.plot(loss_history["epoch"], loss_history[component], 
                        'o-', linewidth=2, 
                        label=component.replace('_', ' ').title(),
                        color=colors.get(component, None))
        
        plt.title('Training Loss', fontsize=15)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.output_dir, "loss_history.png")
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Saved loss history plot to {save_path}")
        return save_path
    
    def visualize_embeddings(self, embeddings, title="Embeddings Visualization", method="pca"):
       
        logger.info(f"Visualizing embeddings using {method}...")
        
        # Stack all embeddings
        stacked_embs = np.vstack([embeddings[k] for k in embeddings])
        
        # Apply dimensionality reduction
        if method == "pca":
            reducer = PCA(n_components=2)
            reduced_embs = reducer.fit_transform(stacked_embs)
            explained_var = reducer.explained_variance_ratio_
            method_name = "PCA"
            subtitle = f"(Explained variance: {explained_var[0]:.2%}, {explained_var[1]:.2%})"
        elif method == "tsne":
            reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
            reduced_embs = reducer.fit_transform(stacked_embs)
            method_name = "t-SNE"
            subtitle = ""
        else:
            raise ValueError(f"Unknown visualization method: {method}")
        
        # Split reduced embeddings back by language
        start_idx = 0
        reduced_by_lang = {}
        for lang, embs in embeddings.items():
            end_idx = start_idx + embs.shape[0]
            reduced_by_lang[lang] = reduced_embs[start_idx:end_idx]
            start_idx = end_idx
        
        # Plotting
        plt.figure(figsize=(10, 8))
        
        # Define colors and markers
        colors = {
            "english": "blue",
            "etok": "green",
            "ktoe": "red",
            "korean": "purple"
        }
        
        markers = {
            "english": "o",
            "etok": "s",
            "ktoe": "^",
            "korean": "D"
        }
        
        # Plot each language variant
        for lang, embs in reduced_by_lang.items():
            plt.scatter(
                embs[:, 0], embs[:, 1],
                color=colors[lang],
                marker=markers[lang],
                label=lang.capitalize(),
                alpha=0.7,
                s=70
            )
        
        plt.title(f"{title}\n{method_name} {subtitle}", fontsize=15)
        plt.xlabel(f"{method_name} Component 1", fontsize=12)
        plt.ylabel(f"{method_name} Component 2", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        return plt
    
    def compare_embeddings(self, original_embeddings, finetuned_embeddings):
        logger.info("Comparing embeddings before and after fine-tuning...")
        
        results = {
            "visualizations": {},
            "distances": {}
        }
        
        # Create PCA visualizations
        fig_pca_orig = self.visualize_embeddings(
            original_embeddings, 
            title="Original Model Embeddings (Before Fine-tuning)",
            method="pca"
        )
        pca_orig_path = os.path.join(self.output_dir, "pca_original.png")
        fig_pca_orig.savefig(pca_orig_path)
        plt.close()
        results["visualizations"]["pca_original"] = pca_orig_path
        
        fig_pca_ft = self.visualize_embeddings(
            finetuned_embeddings, 
            title="Fine-tuned Model Embeddings",
            method="pca"
        )
        pca_ft_path = os.path.join(self.output_dir, "pca_finetuned.png")
        fig_pca_ft.savefig(pca_ft_path)
        plt.close()
        results["visualizations"]["pca_finetuned"] = pca_ft_path
        
        # Create t-SNE visualizations
        fig_tsne_orig = self.visualize_embeddings(
            original_embeddings, 
            title="Original Model Embeddings (Before Fine-tuning)",
            method="tsne"
        )
        tsne_orig_path = os.path.join(self.output_dir, "tsne_original.png")
        fig_tsne_orig.savefig(tsne_orig_path)
        plt.close()
        results["visualizations"]["tsne_original"] = tsne_orig_path
        
        fig_tsne_ft = self.visualize_embeddings(
            finetuned_embeddings, 
            title="Fine-tuned Model Embeddings",
            method="tsne"
        )
        tsne_ft_path = os.path.join(self.output_dir, "tsne_finetuned.png")
        fig_tsne_ft.savefig(tsne_ft_path)
        plt.close()
        results["visualizations"]["tsne_finetuned"] = tsne_ft_path
        
        # Create side-by-side comparison
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Define colors and markers
        colors = {
            "english": "blue",
            "etok": "green",
            "ktoe": "red",
            "korean": "purple"
        }
        
        markers = {
            "english": "o",
            "etok": "s",
            "ktoe": "^",
            "korean": "D"
        }
        
        # PCA Original
        for lang, embs in original_embeddings.items():
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(embs)
            axes[0, 0].scatter(
                reduced[:, 0], reduced[:, 1],
                color=colors[lang],
                marker=markers[lang],
                label=lang.capitalize(),
                alpha=0.7,
                s=70
            )
        axes[0, 0].set_title("Original Model - PCA", fontsize=14)
        axes[0, 0].grid(True, linestyle='--', alpha=0.3)
        axes[0, 0].legend(fontsize=10)
        
        # PCA Fine-tuned
        for lang, embs in finetuned_embeddings.items():
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(embs)
            axes[0, 1].scatter(
                reduced[:, 0], reduced[:, 1],
                color=colors[lang],
                marker=markers[lang],
                label=lang.capitalize(),
                alpha=0.7,
                s=70
            )
        axes[0, 1].set_title("Fine-tuned Model - PCA", fontsize=14)
        axes[0, 1].grid(True, linestyle='--', alpha=0.3)
        axes[0, 1].legend(fontsize=10)
        
        # t-SNE Original
        all_embs_orig = np.vstack([original_embeddings[k] for k in original_embeddings])
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        reduced_orig = tsne.fit_transform(all_embs_orig)
        
        start_idx = 0
        for lang, embs in original_embeddings.items():
            end_idx = start_idx + embs.shape[0]
            axes[1, 0].scatter(
                reduced_orig[start_idx:end_idx, 0], 
                reduced_orig[start_idx:end_idx, 1],
                color=colors[lang],
                marker=markers[lang],
                label=lang.capitalize(),
                alpha=0.7,
                s=70
            )
            start_idx = end_idx
        axes[1, 0].set_title("Original Model - t-SNE", fontsize=14)
        axes[1, 0].grid(True, linestyle='--', alpha=0.3)
        axes[1, 0].legend(fontsize=10)
        
        # t-SNE Fine-tuned
        all_embs_ft = np.vstack([finetuned_embeddings[k] for k in finetuned_embeddings])
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        reduced_ft = tsne.fit_transform(all_embs_ft)
        
        start_idx = 0
        for lang, embs in finetuned_embeddings.items():
            end_idx = start_idx + embs.shape[0]
            axes[1, 1].scatter(
                reduced_ft[start_idx:end_idx, 0], 
                reduced_ft[start_idx:end_idx, 1],
                color=colors[lang],
                marker=markers[lang],
                label=lang.capitalize(),
                alpha=0.7,
                s=70
            )
            start_idx = end_idx
        axes[1, 1].set_title("Fine-tuned Model - t-SNE", fontsize=14)
        axes[1, 1].grid(True, linestyle='--', alpha=0.3)
        axes[1, 1].legend(fontsize=10)
        
        plt.suptitle("Comparing Embeddings Before and After Fine-tuning", fontsize=18)
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        
        comparison_path = os.path.join(self.output_dir, "embedding_comparison.png")
        plt.savefig(comparison_path, dpi=300)
        plt.close()
        results["visualizations"]["comparison"] = comparison_path
        
        # Calculate average distances
        logger.info("Calculating average L2 distances...")
        
        before_distances = {}
        for lang1 in original_embeddings:
            for lang2 in original_embeddings:
                if lang1 < lang2:  # To avoid duplicates
                    avg_dist = np.mean(np.linalg.norm(
                        original_embeddings[lang1] - original_embeddings[lang2], axis=1
                    ))
                    before_distances[f"{lang1}-{lang2}"] = avg_dist
                    logger.info(f"Before: {lang1} to {lang2}: {avg_dist:.4f}")
        
        after_distances = {}
        for lang1 in finetuned_embeddings:
            for lang2 in finetuned_embeddings:
                if lang1 < lang2:  # To avoid duplicates
                    avg_dist = np.mean(np.linalg.norm(
                        finetuned_embeddings[lang1] - finetuned_embeddings[lang2], axis=1
                    ))
                    after_distances[f"{lang1}-{lang2}"] = avg_dist
                    logger.info(f"After: {lang1} to {lang2}: {avg_dist:.4f}")
        
        # Create distance comparison plot
        pairs = []
        distances_before = []
        distances_after = []
        
        for lang1 in original_embeddings:
            for lang2 in original_embeddings:
                if lang1 < lang2:
                    pair = f"{lang1}-{lang2}"
                    pairs.append(pair)
                    distances_before.append(before_distances[pair])
                    distances_after.append(after_distances[pair])
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(pairs))
        width = 0.35
        
        plt.bar(x - width/2, distances_before, width, label='Before Fine-tuning')
        plt.bar(x + width/2, distances_after, width, label='After Fine-tuning')
        
        plt.xlabel('Language Pairs')
        plt.ylabel('Average L2 Distance')
        plt.title('Distance Between Embeddings')
        plt.xticks(x, pairs, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        distances_path = os.path.join(self.output_dir, "distance_comparison.png")
        plt.savefig(distances_path)
        plt.close()
        results["visualizations"]["distances"] = distances_path
        
        # Save distances to CSV
        import pandas as pd
        distance_data = {
            "pair": pairs,
            "before": distances_before,
            "after": distances_after,
            "change": [(after - before) for before, after in zip(distances_before, distances_after)]
        }
        df = pd.DataFrame(distance_data)
        csv_path = os.path.join(self.output_dir, "distances.csv")
        df.to_csv(csv_path, index=False)
        results["distances_csv"] = csv_path
        
        # Store distances data
        results["distances"] = {
            "pairs": pairs,
            "before": distances_before,
            "after": distances_after,
            "change": [(after - before) for before, after in zip(distances_before, distances_after)]
        }
        
        return results