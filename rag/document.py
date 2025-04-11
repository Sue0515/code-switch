import numpy as np
from typing import Optional, Dict, Any, List

class Document:
    
    def __init__(
        self, 
        id: str, 
        content: str, 
        title: Optional[str] = None,
        language: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None
    ):
 
        self.id = id
        self.content = content
        self.title = title
        self.language = language
        self.metadata = metadata or {}
        self.embedding = embedding
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "title": self.title,
            "language": self.language,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        return cls(
            id=data["id"],
            content=data["content"],
            title=data.get("title"),
            language=data.get("language", "unknown"),
            metadata=data.get("metadata", {})
        )
    
    def __repr__(self) -> str:
        """String representation of the document"""
        return f"Document(id={self.id}, title={self.title}, language={self.language})"
    
    @staticmethod
    def batch_to_dict(documents: List['Document']) -> List[Dict[str, Any]]:
        return [doc.to_dict() for doc in documents]
    
    @classmethod
    def batch_from_dict(cls, data_list: List[Dict[str, Any]]) -> List['Document']:
        return [cls.from_dict(data) for data in data_list]