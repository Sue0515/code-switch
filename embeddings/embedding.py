import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def load_model(model_name):
    """
    Load model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def get_embeddings(texts, tokenizer, model, device='cpu', batch_size=8):
    """
    Generate embeddings using the E5 model
    """
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Add prefix for better quality (recommended by E5 model authors)
        inputs = tokenizer(["query: " + text for text in batch_texts], 
                          padding=True, 
                          truncation=True, 
                          return_tensors="pt",
                          max_length=512)
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0]
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            embeddings = embeddings.cpu()
            all_embeddings.append(embeddings.numpy())
    
    return np.vstack(all_embeddings)

def generate_all_embeddings(sentences, tokenizer, model, device='cpu', batch_size=8):
    """
    Generate embeddings for all language types
    """
    embeddings = {}
    for lang_type, texts in sentences.items():
        print(f"Generating embeddings for {lang_type}...")
        embeddings[lang_type] = get_embeddings(texts, tokenizer, model, device=device, batch_size=batch_size)
    
    return embeddings