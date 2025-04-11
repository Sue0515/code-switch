import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name="BAAI/bge-m3", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Create a copy of the model for comparison
        self.original_model = None
    
    def save_original_model(self):
        """Save a copy of the original model for comparison"""
        self.original_model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.original_model.eval()  # Set to evaluation mode
    
    def get_embedding(self, model, input_ids, attention_mask):
        """
        Get embeddings from the model
        
        Args:
            model: Model to use for inference
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Model output tensor from the pooler
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output
    
    def compute_embeddings(self, data_loader, model=None):
        # Use specified model or default to self.model
        model = model or self.model
        
        # Set model to evaluation mode
        model.eval()
        
        embeddings = {
            "english": [],
            "etok": [],
            "ktoe": [],
            "korean": []
        }
        
        with torch.no_grad():
            for batch in data_loader:
                # Move inputs to device
                batch = {k: v.to(self.device) for k, v in batch.items() if k in [
                    "english_input_ids", "english_attention_mask",
                    "etok_input_ids", "etok_attention_mask",
                    "ktoe_input_ids", "ktoe_attention_mask",
                    "korean_input_ids", "korean_attention_mask"
                ]}
                
                # Compute embeddings for each type
                english_emb = self.get_embedding(
                    model, 
                    batch["english_input_ids"], 
                    batch["english_attention_mask"]
                ).cpu().numpy()
                
                etok_emb = self.get_embedding(
                    model, 
                    batch["etok_input_ids"], 
                    batch["etok_attention_mask"]
                ).cpu().numpy()
                
                ktoe_emb = self.get_embedding(
                    model, 
                    batch["ktoe_input_ids"], 
                    batch["ktoe_attention_mask"]
                ).cpu().numpy()
                
                korean_emb = self.get_embedding(
                    model, 
                    batch["korean_input_ids"], 
                    batch["korean_attention_mask"]
                ).cpu().numpy()
                
                # Store embeddings
                embeddings["english"].append(english_emb)
                embeddings["etok"].append(etok_emb)
                embeddings["ktoe"].append(ktoe_emb)
                embeddings["korean"].append(korean_emb)
        
        # Concatenate all batches
        for key in embeddings:
            if embeddings[key]:  # Check if the list is not empty
                embeddings[key] = np.concatenate(embeddings[key], axis=0)
        
        return embeddings
    
    def evaluate_embeddings(self, data_loader):
        if self.original_model is None:
            raise ValueError("Original model not saved. Call save_original_model() first.")
        
        original_embeddings = self.compute_embeddings(data_loader, self.original_model)
        finetuned_embeddings = self.compute_embeddings(data_loader, self.model)
        
        return original_embeddings, finetuned_embeddings
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """
        Load a saved model
        
        Args:
            path: Path to load the model state dict from
            
        Returns:
            Self for chaining
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        return self
    
    def encode_sentences(self, sentences, batch_size=8):
        self.model.eval()
        embeddings = []
        
        # Process in batches
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            ).to(self.device)
            
            # Compute embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.pooler_output.cpu().numpy()
                embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        if embeddings:
            return np.concatenate(embeddings, axis=0)
        
        return np.array([])