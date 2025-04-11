"""
FAISS index for efficient similarity search
"""
import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Union, Tuple
import logging

from .document import Document

logger = logging.getLogger(__name__)

class FaissIndex:
    def __init__(
        self, 
        dimension: int, 
        index_type: str = "flat", 
        metric_type: str = "cosine"
    ):

        self.dimension = dimension
        self.index_type = index_type
        self.metric_type = metric_type
        self.index = self._create_index()
        self.doc_ids = []  # To map index positions to document IDs
        
    def _create_index(self) -> faiss.Index:
        if self.metric_type == "cosine":
            # For cosine similarity, we use L2 distance on normalized vectors
            if self.index_type == "flat":
                return faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatL2(self.dimension)
                nlist = 100  # Number of clusters
                return faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
        elif self.metric_type == "dot":
            # For dot product similarity
            if self.index_type == "flat":
                return faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(self.dimension)
                nlist = 100  # Number of clusters
                return faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")
    
    def add_documents(self, documents: List[Document]) -> None:
        # Check if documents have embeddings
        if any(doc.embedding is None for doc in documents):
            raise ValueError("All documents must have embeddings before adding to index")
        
        # Extract embeddings
        embeddings = np.array([doc.embedding for doc in documents])
        
        # Normalize embeddings if using cosine similarity
        if self.metric_type == "cosine":
            faiss.normalize_L2(embeddings)
        
        # Add to index
        if self.index_type == "ivf" and not self.index.is_trained:
            self.index.train(embeddings)
        
        self.index.add(embeddings)
        
        # Store document IDs
        self.doc_ids.extend([doc.id for doc in documents])
        
        logger.info(f"Added {len(documents)} documents to index, total: {self.index.ntotal}")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search the index for similar embeddings
        """
        # Ensure query embedding is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query if using cosine similarity
        if self.metric_type == "cosine":
            faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Map indices to document IDs
        results = []
        for i, (distances_row, indices_row) in enumerate(zip(distances, indices)):
            query_results = []
            for dist, idx in zip(distances_row, indices_row):
                if idx >= 0 and idx < len(self.doc_ids):  # Valid index
                    query_results.append({
                        "doc_id": self.doc_ids[idx],
                        "distance": float(dist),
                        "index": int(idx)
                    })
            results.append(query_results)
        
        # If only one query, return just its results
        if len(results) == 1:
            return results[0]
        
        return results
    
    def save(self, path: str) -> None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.faiss")
        
        # Save document IDs
        with open(f"{path}.docids.json", "w") as f:
            json.dump(self.doc_ids, f)
        
        # Save metadata
        metadata = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "num_vectors": self.index.ntotal
        }
        with open(f"{path}.meta.json", "w") as f:
            json.dump(metadata, f)
        
        logger.info(f"Index saved to {path}")
    
    def load(self, path: str) -> None:
        # Load FAISS index
        self.index = faiss.read_index(f"{path}.faiss")
        
        # Load document IDs
        with open(f"{path}.docids.json", "r") as f:
            self.doc_ids = json.load(f)
        
        # Load metadata (optional)
        try:
            with open(f"{path}.meta.json", "r") as f:
                metadata = json.load(f)
                self.dimension = metadata.get("dimension", self.dimension)
                self.index_type = metadata.get("index_type", self.index_type)
                self.metric_type = metadata.get("metric_type", self.metric_type)
        except FileNotFoundError:
            pass
        
        logger.info(f"Loaded index from {path} with {self.index.ntotal} vectors")
    
    def __len__(self) -> int:
        return self.index.ntotal