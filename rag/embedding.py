import os
import torch
import numpy as np
from typing import List, Optional, Union, Dict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from .document import Document
import logging

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(
        self, 
        model_name_or_path: str = "BAAI/bge-m3",
        finetuned_model_path: Optional[str] = None,
        finetuned_weights_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        
        self.model_name = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._initialize_model(
            model_name_or_path, 
            finetuned_model_path,
            finetuned_weights_path
        )
        self.dimension = self.get_embedding_dimension()
        
    def _initialize_model(
        self, 
        model_name_or_path: str, 
        finetuned_model_path: Optional[str] = None,
        finetuned_weights_path: Optional[str] = None
    ) -> SentenceTransformer:

        if finetuned_model_path and os.path.exists(finetuned_model_path):
            # Load a saved SentenceTransformer model
            logger.info(f"Loading fine-tuned SentenceTransformer model from {finetuned_model_path}")
            return SentenceTransformer(finetuned_model_path, device=self.device)
        
        elif finetuned_weights_path and os.path.exists(finetuned_weights_path):
            # Initialize from base model + fine-tuned weights
            logger.info(f"Loading base model {model_name_or_path} with fine-tuned weights")
            from sentence_transformers import models
            
            # Load the base model
            word_embedding_model = models.Transformer(model_name_or_path)
            
            # Load fine-tuned weights
            state_dict = torch.load(finetuned_weights_path, map_location=torch.device(self.device))
            word_embedding_model.auto_model.load_state_dict(state_dict)
            
            # Create full SentenceTransformer model
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            
            return model
        
        else:
            # Just load the base model
            logger.info(f"Loading base model {model_name_or_path}")
            return SentenceTransformer(model_name_or_path, device=self.device)
    
    def get_embedding_dimension(self) -> int:

        return self.model.get_sentence_embedding_dimension()
    
    def encode(
        self, 
        texts: List[str], 
        batch_size: int = 32, 
        show_progress: bool = True,
        **kwargs
    ) -> np.ndarray:

        return self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=show_progress,
            convert_to_tensor=False,  # Return numpy for FAISS
            **kwargs
        )
    
    def encode_documents(
        self, 
        documents: List[Document], 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[Document]:
       
        texts = [doc.content for doc in documents]
        
        if show_progress:
            logger.info(f"Encoding {len(texts)} documents")
            
        embeddings = self.encode(
            texts, 
            batch_size=batch_size, 
            show_progress=show_progress
        )
        
        # Update documents with embeddings
        for i, doc in enumerate(documents):
            doc.embedding = embeddings[i]
        
        return documents
    
    def encode_queries(
        self, 
        queries: List[str], 
        batch_size: int = 32
    ) -> np.ndarray:
       
        return self.encode(queries, batch_size=batch_size)
    
    def compute_similarity(
        self, 
        query_embedding: np.ndarray, 
        document_embedding: np.ndarray
    ) -> float:
       
        # Normalize embeddings for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norm = document_embedding / np.linalg.norm(document_embedding)
        
        # Compute cosine similarity
        return float(np.dot(query_norm, doc_norm))
    
    def save(self, path: str) -> None:
       
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
       
        self.model = SentenceTransformer(path, device=self.device)
        self.dimension = self.get_embedding_dimension()
        logger.info(f"Model loaded from {path}")
        
    def detect_language(self, text: str) -> str:
        # Count characters in different scripts
        en_chars = sum(1 for c in text if ord('a') <= ord(c.lower()) <= ord('z'))
        ko_chars = sum(1 for c in text if 0xAC00 <= ord(c) <= 0xD7A3)  # Korean Hangul range
        
        # Calculate ratios
        total_chars = len([c for c in text if c.isalpha()])
        if total_chars == 0:
            return "unknown"
            
        en_ratio = en_chars / total_chars if total_chars > 0 else 0
        ko_ratio = ko_chars / total_chars if total_chars > 0 else 0
        
        # Determine language
        if en_ratio > 0.7:
            return "english"
        elif ko_ratio > 0.7:
            return "korean"
        elif en_ratio > 0.3 and ko_ratio > 0.3:
            return "code-switched"
        else:
            return "unknown"