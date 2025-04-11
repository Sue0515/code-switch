import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm

from .system import RAGSystem

logger = logging.getLogger(__name__)

class RAGEvaluator:
    
    def __init__(
        self, 
        base_system: RAGSystem, 
        finetuned_system: RAGSystem, 
        output_dir: str = "./rag_results",
        visualization: bool = True
    ):
        """
        Initialize the evaluator
        
        Args:
            base_system: Base RAG system
            finetuned_system: Fine-tuned RAG system
            output_dir: Directory to save results
            visualization: Whether to generate visualizations
        """
        self.base_system = base_system
        self.finetuned_system = finetuned_system
        self.output_dir = output_dir
        self.visualization = visualization
        
        self.results = {
            "base": [],
            "finetuned": [],
            "comparison": []
        }
        
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_queries(
        self, 
        queries: List[Dict[str, Any]], 
        top_k: int = 10,
        show_progress: bool = True
    ) -> None:
        logger.info(f"Evaluating {len(queries)} queries...")
        
        # Process queries
        progress_iter = tqdm(queries, desc="Evaluating queries") if show_progress else queries
        for query_info in progress_iter:
            # Process different query types
            query_results = {}
            
            # Process each query type if present
            for query_type in ["English", "Korean", "EtoK", "KtoE"]:
                if query_type in query_info:
                    query_text = query_info[query_type]
                    
                    # Get results from both systems
                    base_results = self.base_system.retrieve(query_text, top_k=top_k)
                    finetuned_results = self.finetuned_system.retrieve(query_text, top_k=top_k)
                    
                    # Calculate language distribution
                    base_lang_dist = self._count_languages(base_results)
                    finetuned_lang_dist = self._count_languages(finetuned_results)
                    
                    # Store results for this query type
                    query_results[query_type] = {
                        "query_text": query_text,
                        "base_results": base_results,
                        "finetuned_results": finetuned_results,
                        "base_lang_dist": base_lang_dist,
                        "finetuned_lang_dist": finetuned_lang_dist
                    }
            
            # Store overall results for all query types
            for query_type, results in query_results.items():
                # Store base results
                self.results["base"].append({
                    "query": results["query_text"],
                    "query_type": query_type,
                    "original_english": query_info.get("English"),
                    "original_korean": query_info.get("Korean"),
                    "results": results["base_results"],
                    "lang_distribution": results["base_lang_dist"]
                })
                
                # Store finetuned results
                self.results["finetuned"].append({
                    "query": results["query_text"],
                    "query_type": query_type,
                    "original_english": query_info.get("English"),
                    "original_korean": query_info.get("Korean"),
                    "results": results["finetuned_results"],
                    "lang_distribution": results["finetuned_lang_dist"]
                })
                
                # Analyze differences for comparison
                base_doc_ids = [r["doc_id"] for r in results["base_results"]]
                finetuned_doc_ids = [r["doc_id"] for r in results["finetuned_results"]]
                common_docs = set(base_doc_ids) & set(finetuned_doc_ids)
                
                # Calculate rankings
                base_ranks = {doc_id: idx for idx, doc_id in enumerate(base_doc_ids)}
                finetuned_ranks = {doc_id: idx for idx, doc_id in enumerate(finetuned_doc_ids)}
                
                # Calculate rank changes for common documents
                rank_changes = []
                for doc_id in common_docs:
                    base_rank = base_ranks[doc_id] + 1  # 1-indexed
                    finetuned_rank = finetuned_ranks[doc_id] + 1  # 1-indexed
                    rank_changes.append({
                        "doc_id": doc_id,
                        "base_rank": base_rank,
                        "finetuned_rank": finetuned_rank,
                        "change": base_rank - finetuned_rank  # Positive: improved rank
                    })
                
                # Store comparison results
                self.results["comparison"].append({
                    "query": results["query_text"],
                    "query_type": query_type,
                    "original_english": query_info.get("English"),
                    "original_korean": query_info.get("Korean"),
                    "common_docs": len(common_docs),
                    "base_only": len(set(base_doc_ids) - common_docs),
                    "finetuned_only": len(set(finetuned_doc_ids) - common_docs),
                    "rank_changes": rank_changes,
                    "base_lang_distribution": results["base_lang_dist"],
                    "finetuned_lang_distribution": results["finetuned_lang_dist"]
                })
        
        logger.info(f"Evaluation complete for {len(queries)} queries across multiple query types")
    
    def _count_languages(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        langs = {}
        for res in results:
            lang = res.get("language", "unknown")
            langs[lang] = langs.get(lang, 0) + 1
        return langs
    
    def get_query_type_stats(self) -> Dict[str, Dict[str, Any]]:
        query_types = {}
        
        # Group by query type
        for i, comp in enumerate(self.results["comparison"]):
            query_type = comp["query_type"]
            if query_type not in query_types:
                query_types[query_type] = {
                    "queries": [],
                    "base_lang_dist": {"english": 0, "korean": 0, "unknown": 0},
                    "finetuned_lang_dist": {"english": 0, "korean": 0, "unknown": 0},
                    "rank_improvements": 0,
                    "rank_deteriorations": 0,
                    "unchanged": 0
                }
            
            query_types[query_type]["queries"].append(i)
            
            # Update language distributions
            for lang, count in comp["base_lang_distribution"].items():
                query_types[query_type]["base_lang_dist"][lang] = query_types[query_type]["base_lang_dist"].get(lang, 0) + count
            
            for lang, count in comp["finetuned_lang_distribution"].items():
                query_types[query_type]["finetuned_lang_dist"][lang] = query_types[query_type]["finetuned_lang_dist"].get(lang, 0) + count
            
            # Count rank changes
            for change in comp["rank_changes"]:
                if change["change"] > 0:  # Rank improved (lower number = better rank)
                    query_types[query_type]["rank_improvements"] += 1
                elif change["change"] < 0:  # Rank deteriorated
                    query_types[query_type]["rank_deteriorations"] += 1
                else:  # Rank unchanged
                    query_types[query_type]["unchanged"] += 1
        
        return query_types
    
    def calculate_metrics(self) -> Dict[str, Dict[str, float]]:
        metrics = {
            "base": {
                "mrr": 0.0,
                "precision@5": 0.0,
                "ndcg@10": 0.0
            },
            "finetuned": {
                "mrr": 0.0,
                "precision@5": 0.0,
                "ndcg@10": 0.0
            }
        }
        
        return metrics
    
    def plot_language_distributions(self) -> str:
        if not self.visualization:
            return "Visualization disabled"
            
        # Aggregate language distributions for each query type
        query_types = self.get_query_type_stats()
        
        # Plot language distribution by query type
        plt.figure(figsize=(14, 8))
        
        query_type_names = list(query_types.keys())
        x = np.arange(len(query_type_names))
        width = 0.2
        
        # Extract data for plotting
        base_english = [query_types[qt]["base_lang_dist"].get("english", 0) for qt in query_type_names]
        base_korean = [query_types[qt]["base_lang_dist"].get("korean", 0) for qt in query_type_names]
        finetuned_english = [query_types[qt]["finetuned_lang_dist"].get("english", 0) for qt in query_type_names]
        finetuned_korean = [query_types[qt]["finetuned_lang_dist"].get("korean", 0) for qt in query_type_names]
        
        plt.bar(x - width*1.5, base_english, width, label='Base Model (English)', color='skyblue')
        plt.bar(x - width/2, base_korean, width, label='Base Model (Korean)', color='lightgreen')
        plt.bar(x + width/2, finetuned_english, width, label='Fine-tuned Model (English)', color='blue')
        plt.bar(x + width*1.5, finetuned_korean, width, label='Fine-tuned Model (Korean)', color='green')
        
        plt.xlabel('Query Type')
        plt.ylabel('Number of Documents')
        plt.title('Language Distribution by Query Type')
        plt.xticks(x, query_type_names)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.output_dir, "language_distribution_by_query_type.png")
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Language distribution plot saved to {plot_path}")
        return plot_path
    
    def plot_rank_changes(self) -> str:
        if not self.visualization:
            return "Visualization disabled"
            
        # Collect rank changes by query type
        query_types = {}
        for comp in self.results["comparison"]:
            query_type = comp["query_type"]
            if query_type not in query_types:
                query_types[query_type] = []
            
            for change in comp["rank_changes"]:
                query_types[query_type].append(change["change"])
        
        # Plot rank changes by query type
        plt.figure(figsize=(12, 8))
        
        # Create a subplot for each query type
        num_types = len(query_types)
        cols = min(2, num_types)
        rows = (num_types + cols - 1) // cols
        
        for i, (query_type, changes) in enumerate(query_types.items(), 1):
            plt.subplot(rows, cols, i)
            
            if changes:  # Only plot if we have data
                plt.hist(changes, bins=range(min(changes)-1, max(changes)+2), alpha=0.75)
                plt.axvline(x=0, color='r', linestyle='--')
                plt.title(f'Rank Changes for {query_type} Queries')
                plt.xlabel('Rank Change (Positive = Improvement)')
                plt.ylabel('Count')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, "No common documents", ha='center', va='center')
                plt.title(f'No Data for {query_type} Queries')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.output_dir, "rank_changes_by_query_type.png")
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Rank changes plot saved to {plot_path}")
        return plot_path
    
    def generate_report(self) -> str:
        # Calculate overall statistics
        total_queries = len(self.results["comparison"])
        total_common_docs = sum(comp["common_docs"] for comp in self.results["comparison"])
        total_base_only = sum(comp["base_only"] for comp in self.results["comparison"])
        total_finetuned_only = sum(comp["finetuned_only"] for comp in self.results["comparison"])
        
        total_rank_improvements = sum(
            len([c for c in comp["rank_changes"] if c["change"] > 0])
            for comp in self.results["comparison"])
        
        total_rank_deteriorations = sum(
            len([c for c in comp["rank_changes"] if c["change"] < 0])
            for comp in self.results["comparison"])
        
        # Calculate language distributions
        base_langs = {"english": 0, "korean": 0, "unknown": 0}
        finetuned_langs = {"english": 0, "korean": 0, "unknown": 0}
        
        for result in self.results["base"]:
            for lang, count in result["lang_distribution"].items():
                base_langs[lang] = base_langs.get(lang, 0) + count
        
        for result in self.results["finetuned"]:
            for lang, count in result["lang_distribution"].items():
                finetuned_langs[lang] = finetuned_langs.get(lang, 0) + count
        
        # Generate query type statistics
        query_type_stats = self.get_query_type_stats()
        
        # Generate and save visualizations
        plot_paths = {}
        if self.visualization:
            plot_paths["language_distribution"] = self.plot_language_distributions()
            plot_paths["rank_changes"] = self.plot_rank_changes()
        
        # Generate Markdown report
        report = f"""# RAG Evaluation Report for Code-Switched Queries

                    ## Overview

                    - Total Queries: {total_queries}
                    - Results with Common Documents: {total_common_docs}
                    - Documents Found Only by Base Model: {total_base_only}
                    - Documents Found Only by Fine-tuned Model: {total_finetuned_only}
                    - Rank Improvements: {total_rank_improvements}
                    - Rank Deteriorations: {total_rank_deteriorations}

                    ## Language Distribution

                    ### Base Model
                    {self._format_distribution(base_langs)}

                    ### Fine-tuned Model
                    {self._format_distribution(finetuned_langs)}

                    ## Results by Query Type

                    """
        # Add query type statistics
        for query_type, stats in query_type_stats.items():
            report += f"""### {query_type}
                        - Queries: {len(stats["queries"])}
                        - Base Model Language Distribution: {self._format_distribution(stats["base_lang_dist"])}
                        - Fine-tuned Model Language Distribution: {self._format_distribution(stats["finetuned_lang_dist"])}
                        - Rank Improvements: {stats["rank_improvements"]}
                        - Rank Deteriorations: {stats["rank_deteriorations"]}
                        - Unchanged Ranks: {stats["unchanged"]}

                        """
        # Save report
        report_path = os.path.join(self.output_dir, "evaluation_report.md")
        with open(report_path, "w") as f:
            f.write(report)
        
        # Save detailed results
        results_path = os.path.join(self.output_dir, "detailed_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Report generated at {report_path}")
        
        return report_path
    
    def _format_distribution(self, dist: Dict[str, int]) -> str:

        total = sum(dist.values())
        if total == 0:
            return "N/A"
        
        parts = []
        for lang, count in dist.items():
            percent = (count / total) * 100 if total > 0 else 0
            parts.append(f"{lang}: {count} ({percent:.1f}%)")
        
        return ", ".join(parts)
    
    def export_results_to_csv(self) -> str:
        # Create DataFrame for comparison results
        rows = []
        
        for comp in self.results["comparison"]:
            # Basic query info
            row = {
                "query": comp["query"],
                "query_type": comp["query_type"],
                "common_docs": comp["common_docs"],
                "base_only": comp["base_only"],
                "finetuned_only": comp["finetuned_only"]
            }
            
            # Language distribution
            for lang, count in comp["base_lang_distribution"].items():
                row[f"base_{lang}"] = count
            
            for lang, count in comp["finetuned_lang_distribution"].items():
                row[f"finetuned_{lang}"] = count
            
            # Rank change summary
            improvements = len([c for c in comp["rank_changes"] if c["change"] > 0])
            deteriorations = len([c for c in comp["rank_changes"] if c["change"] < 0])
            unchanged = len([c for c in comp["rank_changes"] if c["change"] == 0])
            
            row["rank_improvements"] = improvements
            row["rank_deteriorations"] = deteriorations
            row["rank_unchanged"] = unchanged
            
            rows.append(row)
        
        # Create DataFrame and export to CSV
        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.output_dir, "evaluation_results.csv")
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Results exported to {csv_path}")
        return csv_path