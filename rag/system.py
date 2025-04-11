"""
Complete RAG system for multilingual and code-switched queries
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm

from .document import Document
from .embedding import EmbeddingModel
from .index import FaissIndex
from .retriever import WikipediaRetriever

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        documents: Optional[List[Document]] = None,
        index: Optional[FaissIndex] = None,
        name: str = "default",
        metric_type: str = "cosine"
    ):
        """
        Initialize the RAG system
        
        Args:
            embedding_model: EmbeddingModel for encoding queries and documents
            documents: Initial list of documents (optional)
            index: FaissIndex for document retrieval (optional)
            name: Name identifier for the system
            metric_type: Type of similarity metric to use
        """
        self.embedding_model = embedding_model
        self.documents = documents or []
        self.document_map = {}  # Maps document IDs to documents
        self.index = index
        self.name = name
        
        # Initialize document map
        for doc in self.documents:
            self.document_map[doc.id] = doc
        
        # Create index if not provided
        if self.index is None and self.embedding_model is not None:
            self.index = FaissIndex(
                self.embedding_model.dimension, 
                metric_type=metric_type
            )
    
    def add_documents(
        self, 
        documents: List[Document], 
        batch_size: int = 32, 
        update_index: bool = True
    ) -> None:
        # Store documents in map
        for doc in documents:
            self.document_map[doc.id] = doc
        
        # Compute embeddings if not already present
        docs_to_embed = [doc for doc in documents if doc.embedding is None]
        if docs_to_embed and self.embedding_model:
            logger.info(f"Computing embeddings for {len(docs_to_embed)} documents")
            self.embedding_model.encode_documents(docs_to_embed, batch_size=batch_size)
        
        # Add to index if requested
        if update_index and self.index:
            self.index.add_documents(documents)
        
        # Update document list
        self.documents.extend(documents)
        
        logger.info(f"Added {len(documents)} documents to system, total: {len(self.documents)}")
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search index
        search_results = self.index.search(query_embedding, k=top_k)
        
        # Format results
        results = []
        for i, result in enumerate(search_results):
            doc_id = result["doc_id"]
            doc = self.document_map.get(doc_id)
            
            if doc:
                results.append({
                    "rank": i + 1,
                    "doc_id": doc_id,
                    "title": doc.title,
                    "content": doc.content,
                    "language": doc.language,
                    "distance": result["distance"],
                    "score": self._convert_distance_to_score(result["distance"]),
                    "metadata": doc.metadata
                })
        
        return results
    
    def _convert_distance_to_score(self, distance: float) -> float:
        """
        Convert distance to similarity score
        
        Args:
            distance: Distance value from FAISS index
            
        Returns:
            Similarity score (higher is more similar)
        """
        # For cosine distance (L2 on normalized vectors), convert to similarity
        return 1.0 - (distance / 2.0)
    
    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save index
        self.index.save(os.path.join(output_dir, f"{self.name}_index"))
        
        # Save documents
        documents_data = [doc.to_dict() for doc in self.documents]
        documents_path = os.path.join(output_dir, f"{self.name}_documents.json")
        
        with open(documents_path, "w", encoding="utf-8") as f:
            json.dump(documents_data, f, ensure_ascii=False, indent=2)
        
        # Save metadata
        metadata = {
            "name": self.name,
            "num_documents": len(self.documents),
            "embedding_model": self.embedding_model.model_name,
            "embedding_dimension": self.embedding_model.dimension
        }
        
        metadata_path = os.path.join(output_dir, f"{self.name}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved RAG system '{self.name}' to {output_dir}")
    
    def load(self, output_dir: str, embedding_model: Optional[EmbeddingModel] = None) -> None:
        # Update embedding model if provided
        if embedding_model:
            self.embedding_model = embedding_model
        
        # Load index
        self.index.load(os.path.join(output_dir, f"{self.name}_index"))
        
        # Load documents
        documents_path = os.path.join(output_dir, f"{self.name}_documents.json")
        with open(documents_path, "r", encoding="utf-8") as f:
            documents_data = json.load(f)
        
        # Create document objects
        self.documents = []
        self.document_map = {}
        for doc_data in documents_data:
            doc = Document.from_dict(doc_data)
            self.documents.append(doc)
            self.document_map[doc.id] = doc
        
        logger.info(f"Loaded RAG system '{self.name}' from {output_dir} with {len(self.documents)} documents")
    
    @classmethod
    def build(
        cls,
        embedding_model: EmbeddingModel,
        queries: List[Union[str, Dict[str, Any]]],
        wikipedia_langs: List[str] = ["en", "ko"],
        documents_per_query: int = 5,
        name: str = "default",
        metric_type: str = "cosine"
    ) -> 'RAGSystem':
        
        # Create retriever and build corpus
        retriever = WikipediaRetriever(langs=wikipedia_langs)
        corpus = retriever.build_corpus(
            queries=queries,
            limit_per_query=documents_per_query
        )
        
        # Create RAG system
        rag_system = cls(
            embedding_model=embedding_model,
            name=name,
            metric_type=metric_type
        )
        
        # Add documents to system
        rag_system.add_documents(corpus)
        
        return rag_system
    
    def process_query(
        self, 
        query: str, 
        top_k: int = 5
    ) -> Dict[str, Any]:
        
        # Detect language
        query_language = self.embedding_model.detect_language(query)
        
        # Retrieve documents
        retrieval_results = self.retrieve(query, top_k=top_k)
        
        # Count languages in results
        language_counts = {}
        for res in retrieval_results:
            lang = res.get("language", "unknown")
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        # Return complete results
        return {
            "query": query,
            "language": query_language,
            "retrieved_documents": retrieval_results,
            "language_distribution": language_counts
        }