import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class RAGConfig:
    """Configuration for RAG pipeline"""
    
    # Basic paths
    query_file: str = "data/code_switch_queries.json"
    output_dir: str = "./rag_results"
    
    # Embedding model settings
    base_model_name: str = "BAAI/bge-m3"
    finetuned_model_path: Optional[str] = None
    finetuned_weights_path: Optional[str] = None
    device: Optional[str] = None  # Will default to CUDA if available
    
    # Corpus settings
    use_wikipedia: bool = True
    wikipedia_langs: List[str] = field(default_factory=lambda: ["en", "ko"])
    documents_per_query: int = 5
    corpus_path: Optional[str] = None
    save_corpus: bool = True
    
    # Retrieval settings
    index_type: str = "flat"  # "flat" or "ivf"
    metric_type: str = "cosine"  # "cosine" or "dot"
    top_k: int = 10
    
    # Generation settings (placeholder for future integration)
    llm_model_name: Optional[str] = None
    
    # Evaluation settings
    visualization: bool = True
    evaluation_metrics: List[str] = field(default_factory=lambda: ["mrr", "ndcg", "precision"])
    
    def save(self, path: str) -> None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Convert to dictionary
        config_dict = self.__dict__.copy()
        
        # Save to file
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Configuration saved to {path}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, path: str) -> 'RAGConfig':
        with open(path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)