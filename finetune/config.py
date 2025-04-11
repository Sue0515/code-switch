import os
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class FinetuningConfig:
    """Configuration for fine-tuning multilingual embedding models"""
    
    # Data settings
    data_file: str = "data/code-switch.json"
    output_dir: str = "results"
    
    # Model settings
    model_name: str = "BAAI/bge-m3"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Training settings
    num_epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-5
    max_length: int = 128
    
    # Loss settings
    loss_type: str = "refined"  # We'll only use "refined"
    temperature: float = 0.1
    lambda_cs_reg: float = 0.5
    
    # Evaluation settings
    eval_batch_size: int = 8
    visualize_embeddings: bool = True
    
    def save(self, path):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            # Convert to dict and save as JSON
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def from_json(cls, path):
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)