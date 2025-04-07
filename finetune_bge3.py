import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import json
import datetime
import random
import os

# Define the dataset class for handling code-switch.json
class MultilingualDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=128, calculate_cs_ratio=True):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.calculate_cs_ratio = calculate_cs_ratio
        
        # Calculate code-switching ratios if requested
        if calculate_cs_ratio:
            self.cs_ratios = self._compute_cs_ratios()
        else:
            self.cs_ratios = None

    def __len__(self):
        return len(self.data)
    
    def _compute_cs_ratios(self):
        """
        Compute code-switching ratios for each example.
        Returns a list of tuples (etok_ratio, ktoe_ratio) where:
        - etok_ratio: How "English" the EtoK example is (0.0 to 1.0)
        - ktoe_ratio: How "Korean" the KtoE example is (0.0 to 1.0)
        """
        cs_ratios = []
        for item in self.data:
            # Simple heuristic: Count English characters vs Korean characters
            # For EtoK: higher ratio = more English content
            etok_english_chars = sum(1 for c in item["EtoK"] if ord(c) < 128)
            etok_total_chars = len(item["EtoK"].replace(" ", ""))
            etok_ratio = etok_english_chars / etok_total_chars if etok_total_chars > 0 else 0.5
            
            # For KtoE: higher ratio = more Korean content
            ktoe_english_chars = sum(1 for c in item["KtoE"] if ord(c) < 128)
            ktoe_total_chars = len(item["KtoE"].replace(" ", ""))
            ktoe_ratio = 1 - (ktoe_english_chars / ktoe_total_chars if ktoe_total_chars > 0 else 0.5)
            
            cs_ratios.append((etok_ratio, ktoe_ratio))
        
        return cs_ratios

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize all versions of the sentence (English, EtoK, KtoE, Korean)
        english = self.tokenizer(item["English"], padding="max_length", truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
        etok = self.tokenizer(item["EtoK"], padding="max_length", truncation=True,
                              max_length=self.max_length, return_tensors="pt")
        ktoe = self.tokenizer(item["KtoE"], padding="max_length", truncation=True,
                              max_length=self.max_length, return_tensors="pt")
        korean = self.tokenizer(item["Korean"], padding="max_length", truncation=True,
                                max_length=self.max_length, return_tensors="pt")

        result = {
            "english_input_ids": english["input_ids"].squeeze(0),
            "english_attention_mask": english["attention_mask"].squeeze(0),
            "etok_input_ids": etok["input_ids"].squeeze(0),
            "etok_attention_mask": etok["attention_mask"].squeeze(0),
            "ktoe_input_ids": ktoe["input_ids"].squeeze(0),
            "ktoe_attention_mask": ktoe["attention_mask"].squeeze(0),
            "korean_input_ids": korean["input_ids"].squeeze(0),
            "korean_attention_mask": korean["attention_mask"].squeeze(0)
        }
        
        # Add code-switching ratios if available
        if self.cs_ratios:
            result["etok_ratio"] = torch.tensor(self.cs_ratios[idx][0], dtype=torch.float)
            result["ktoe_ratio"] = torch.tensor(self.cs_ratios[idx][1], dtype=torch.float)
        
        return result

# Define custom loss functions
class CustomLoss(nn.Module):
    def __init__(self, temperature=0.1, lambda_cohesion=1.0, lambda_separation=1.0):
        super(CustomLoss, self).__init__()
        self.temperature = temperature
        self.lambda_cohesion = lambda_cohesion
        self.lambda_separation = lambda_separation

    def forward(self, embeddings_dict):
        """
        Args:
            embeddings_dict: Dictionary containing embeddings for English, EtoK, KtoE, and Korean.
                            Keys: 'english', 'etok', 'ktoe', 'korean'.
                            Values: Tensors of shape (batch_size, embedding_dim).
        Returns:
            Total loss value.
        """
        # Extract embeddings for each category
        english_emb = embeddings_dict["english"]
        etok_emb = embeddings_dict["etok"]
        ktoe_emb = embeddings_dict["ktoe"]
        korean_emb = embeddings_dict["korean"]

        # Normalize embeddings for cosine similarity
        english_emb = nn.functional.normalize(english_emb, p=2, dim=1)
        etok_emb = nn.functional.normalize(etok_emb, p=2, dim=1)
        ktoe_emb = nn.functional.normalize(ktoe_emb, p=2, dim=1)
        korean_emb = nn.functional.normalize(korean_emb, p=2, dim=1)

        # Contrastive Loss (for semantic alignment)
        positive_pairs = [
            (english_emb, etok_emb),
            (english_emb, ktoe_emb),
            (english_emb, korean_emb)
        ]
        
        contrastive_loss = 0.0
        for anchor_emb, positive_emb in positive_pairs:
            similarity_matrix = torch.mm(anchor_emb, positive_emb.T) / self.temperature
            contrastive_loss += -torch.log(torch.exp(similarity_matrix.diag()) /
                                           torch.exp(similarity_matrix).sum(dim=1)).mean()

        # Cluster Center Loss (cohesion within clusters)
        cluster_centers = {
            "english": english_emb.mean(dim=0),
            "etok": etok_emb.mean(dim=0),
            "ktoe": ktoe_emb.mean(dim=0),
            "korean": korean_emb.mean(dim=0)
        }
        
        cohesion_loss = 0.0
        for key in cluster_centers:
            cluster_center = cluster_centers[key]
            cohesion_loss += torch.norm(embeddings_dict[key] - cluster_center.unsqueeze(0), dim=1).mean()

        # Inter-Cluster Separation Loss (separation between clusters)
        separation_loss = 0.0
        cluster_center_values = list(cluster_centers.values())
        
        for i in range(len(cluster_center_values)):
            for j in range(i + 1, len(cluster_center_values)):
                separation_loss += -torch.norm(cluster_center_values[i] - cluster_center_values[j])

        separation_loss /= len(cluster_center_values) * (len(cluster_center_values) - 1) / 2

        # Combine losses with weights
        total_loss = contrastive_loss + self.lambda_cohesion * cohesion_loss + self.lambda_separation * separation_loss
        
        # Return loss components for analysis
        return total_loss, {
            "contrastive_loss": contrastive_loss.item(),
            "cohesion_loss": cohesion_loss.item(),
            "separation_loss": separation_loss.item(),
            "total_loss": total_loss.item()
        }

class CodeSwitchLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(CodeSwitchLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, embeddings_dict, cs_ratios=None):
        """
        Args:
            embeddings_dict: Dictionary with embeddings
            cs_ratios: Tuple of tensors (etok_ratio, ktoe_ratio) of shape (batch_size,)
                       indicating language ratios in code-switched examples
        """
        # Normalize embeddings
        english_emb = nn.functional.normalize(embeddings_dict["english"], p=2, dim=1)
        etok_emb = nn.functional.normalize(embeddings_dict["etok"], p=2, dim=1)
        ktoe_emb = nn.functional.normalize(embeddings_dict["ktoe"], p=2, dim=1)
        korean_emb = nn.functional.normalize(embeddings_dict["korean"], p=2, dim=1)
        
        batch_size = english_emb.shape[0]
        
        # 1. Basic contrastive loss
        contrastive_loss = 0.0
        for i in range(batch_size):
            # Calculate similarities between this example's different language versions
            e = english_emb[i:i+1]  # Shape: (1, dim)
            k = korean_emb[i:i+1]
            etk = etok_emb[i:i+1]
            kte = ktoe_emb[i:i+1]
            
            # Stack all embeddings from other examples as negatives
            # (excluding this example's different language versions)
            negatives = []
            for j in range(batch_size):
                if j != i:
                    negatives.append(english_emb[j:j+1])
                    negatives.append(korean_emb[j:j+1])
                    negatives.append(etok_emb[j:j+1])
                    negatives.append(ktoe_emb[j:j+1])
            
            if negatives:  # Make sure we have at least one negative
                neg_stack = torch.cat(negatives, dim=0)  # Shape: (batch_size*4-4, dim)
                
                # For each language version, compute loss against all others
                versions = [e, k, etk, kte]
                for idx, anchor in enumerate(versions):
                    # All other versions of this example are positives
                    positives = torch.cat([v for i, v in enumerate(versions) if i != idx], dim=0)
                    
                    # Compute similarities
                    pos_sim = torch.mm(anchor, positives.t()) / self.temperature  # (1, 3)
                    neg_sim = torch.mm(anchor, neg_stack.t()) / self.temperature  # (1, batch_size*4-4)
                    
                    # InfoNCE loss
                    numerator = torch.exp(pos_sim).sum()
                    denominator = numerator + torch.exp(neg_sim).sum()
                    contrastive_loss += -torch.log(numerator / denominator)
        
        # 2. Code-switching aware regularization (new)
        cs_reg_loss = 0.0
        if cs_ratios is not None:
            etok_ratios, ktoe_ratios = cs_ratios
            
            for i in range(batch_size):
                etok_ratio = etok_ratios[i]
                ktoe_ratio = ktoe_ratios[i]
                
                # EtoK should be etok_ratio% similar to English, (1-etok_ratio)% similar to Korean
                etok_target = etok_ratio * english_emb[i] + (1 - etok_ratio) * korean_emb[i]
                etok_reg = torch.norm(etok_emb[i] - etok_target)
                
                # KtoE should be ktoe_ratio% similar to Korean, (1-ktoe_ratio)% similar to English
                ktoe_target = ktoe_ratio * korean_emb[i] + (1 - ktoe_ratio) * english_emb[i]
                ktoe_reg = torch.norm(ktoe_emb[i] - ktoe_target)
                
                cs_reg_loss += etok_reg + ktoe_reg
        
        # Combine losses
        total_loss = contrastive_loss + 0.5 * cs_reg_loss
        
        # Return loss components for analysis
        components = {
            "contrastive_loss": float(contrastive_loss) if isinstance(contrastive_loss, torch.Tensor) else contrastive_loss,
            "cs_reg_loss": float(cs_reg_loss) if isinstance(cs_reg_loss, torch.Tensor) else cs_reg_loss,
            "total_loss": float(total_loss) if isinstance(total_loss, torch.Tensor) else total_loss
        }

class RefinedCodeSwitchLoss(nn.Module):
    def __init__(self, temperature=0.1, lambda_cs_reg=0.5):
        super(RefinedCodeSwitchLoss, self).__init__()
        self.temperature = temperature
        self.lambda_cs_reg = lambda_cs_reg
        
    def forward(self, embeddings_dict, cs_ratios=None):
        """
        Args:
            embeddings_dict: Dictionary with embeddings
            cs_ratios: Tuple of tensors (etok_ratio, ktoe_ratio) of shape (batch_size,)
                       indicating language ratios in code-switched examples
        Returns:
            total_loss: Combined loss value
            components: Dictionary with individual loss components for tracking
        """
        # Normalize embeddings
        english_emb = nn.functional.normalize(embeddings_dict["english"], p=2, dim=1)
        etok_emb = nn.functional.normalize(embeddings_dict["etok"], p=2, dim=1)
        ktoe_emb = nn.functional.normalize(embeddings_dict["ktoe"], p=2, dim=1)
        korean_emb = nn.functional.normalize(embeddings_dict["korean"], p=2, dim=1)
        
        batch_size = english_emb.shape[0]
        
        # 1. Contrastive loss using only English and Korean as anchors
        contrastive_loss = 0.0
        
        # Process each example in the batch
        for i in range(batch_size):
            # English as anchor
            eng_anchor = english_emb[i:i+1]  # Shape: (1, dim)
            
            # Positives for English: same-content EtoK, KtoE, Korean
            eng_positives = torch.cat([
                etok_emb[i:i+1],
                ktoe_emb[i:i+1], 
                korean_emb[i:i+1]
            ], dim=0)  # Shape: (3, dim)
            
            # Korean as anchor
            kor_anchor = korean_emb[i:i+1]  # Shape: (1, dim)
            
            # Positives for Korean: same-content EtoK, KtoE, English
            kor_positives = torch.cat([
                etok_emb[i:i+1],
                ktoe_emb[i:i+1], 
                english_emb[i:i+1]
            ], dim=0)  # Shape: (3, dim)
            
            # Collect negatives: all examples from other items in batch
            negatives = []
            for j in range(batch_size):
                if j != i:
                    negatives.append(english_emb[j:j+1])
                    negatives.append(korean_emb[j:j+1])
                    negatives.append(etok_emb[j:j+1])
                    negatives.append(ktoe_emb[j:j+1])
            
            if negatives:  # Make sure we have at least one negative
                neg_stack = torch.cat(negatives, dim=0)  # Shape: (batch_size*4-4, dim)
                
                # English anchor with its positives and negatives
                eng_pos_sim = torch.mm(eng_anchor, eng_positives.t()) / self.temperature  # (1, 3)
                eng_neg_sim = torch.mm(eng_anchor, neg_stack.t()) / self.temperature  # (1, batch_size*4-4)
                
                # InfoNCE loss for English anchor
                eng_numerator = torch.exp(eng_pos_sim).sum()
                eng_denominator = eng_numerator + torch.exp(eng_neg_sim).sum()
                contrastive_loss += -torch.log(eng_numerator / eng_denominator)
                
                # Korean anchor with its positives and negatives
                kor_pos_sim = torch.mm(kor_anchor, kor_positives.t()) / self.temperature  # (1, 3)
                kor_neg_sim = torch.mm(kor_anchor, neg_stack.t()) / self.temperature  # (1, batch_size*4-4)
                
                # InfoNCE loss for Korean anchor
                kor_numerator = torch.exp(kor_pos_sim).sum()
                kor_denominator = kor_numerator + torch.exp(kor_neg_sim).sum()
                contrastive_loss += -torch.log(kor_numerator / kor_denominator)
        
        # 2. Code-switching aware regularization
        cs_reg_loss = 0.0
        if cs_ratios is not None:
            etok_ratios, ktoe_ratios = cs_ratios
            
            for i in range(batch_size):
                etok_ratio = etok_ratios[i]
                ktoe_ratio = ktoe_ratios[i]
                
                # EtoK should be etok_ratio% similar to English, (1-etok_ratio)% similar to Korean
                etok_target = etok_ratio * english_emb[i] + (1 - etok_ratio) * korean_emb[i]
                etok_reg = torch.norm(etok_emb[i] - etok_target)
                
                # KtoE should be ktoe_ratio% similar to Korean, (1-ktoe_ratio)% similar to English
                ktoe_target = ktoe_ratio * korean_emb[i] + (1 - ktoe_ratio) * english_emb[i]
                ktoe_reg = torch.norm(ktoe_emb[i] - ktoe_target)
                
                cs_reg_loss += etok_reg + ktoe_reg
        
        # Normalize losses by batch size
        contrastive_loss = contrastive_loss / (2 * batch_size)  # 2 anchors per example
        cs_reg_loss = cs_reg_loss / batch_size if cs_ratios is not None else 0.0
        
        # Combine losses
        total_loss = contrastive_loss + self.lambda_cs_reg * cs_reg_loss
        
        # Return both total loss and components for tracking
        components = {
            "contrastive_loss": float(contrastive_loss) if isinstance(contrastive_loss, torch.Tensor) else contrastive_loss,
            "cs_reg_loss": float(cs_reg_loss) if isinstance(cs_reg_loss, torch.Tensor) else cs_reg_loss,
            "total_loss": float(total_loss) if isinstance(total_loss, torch.Tensor) else total_loss
        }
        
        return total_loss, components

# Define the finetuning class
class EmbeddingFinetuner:
    def __init__(self, model_name="BAAI/bge-m3", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

        # Save the original model for comparison
        self.original_model = AutoModel.from_pretrained(model_name).to(device)
        self.original_model.eval()  # Set to evaluation mode
        
        # Create directory for results
        self.results_dir = f"results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize loss tracking
        self.loss_history = {
            "epoch": [],
            "total_loss": [],
            "contrastive_loss": [],
            "cs_reg_loss": [],
            # "cohesion_loss": [],
            # "separation_loss": []
        }
    
    def get_embedding(self, model, input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output

    def finetune(self, train_dataloader, loss_type="refined", num_epochs=5, lr=1e-5):
        """
        Fine-tune the model with specified loss function
        
        Args:
            train_dataloader: DataLoader for training data
            loss_type: Type of loss function to use ('custom', 'codeswitch', or 'refined')
            num_epochs: Number of training epochs
            lr: Learning rate
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        # Select loss function
        if loss_type == "custom":
            loss_fn = CustomLoss()
            print("Using CustomLoss")
        elif loss_type == "codeswitch":
            loss_fn = CodeSwitchLoss()
            print("Using CodeSwitchLoss")
        elif loss_type == "refined":
            loss_fn = RefinedCodeSwitchLoss()
            print("Using RefinedCodeSwitchLoss")
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        self.model.train()
        
        # Track best model
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_total_loss = 0.0
            epoch_loss_components = {
                "contrastive_loss": 0.0,
                "cs_reg_loss": 0.0,
                "cohesion_loss": 0.0,
                "separation_loss": 0.0
            }
            
            for batch in train_dataloader:
                optimizer.zero_grad()

                # Move inputs to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Compute embeddings for all categories
                embeddings_dict = {
                    "english": self.get_embedding(self.model, batch["english_input_ids"], batch["english_attention_mask"]),
                    "etok": self.get_embedding(self.model, batch["etok_input_ids"], batch["etok_attention_mask"]),
                    "ktoe": self.get_embedding(self.model, batch["ktoe_input_ids"], batch["ktoe_attention_mask"]),
                    "korean": self.get_embedding(self.model, batch["korean_input_ids"], batch["korean_attention_mask"])
                }
                
                # Extract code-switching ratios if available
                cs_ratios = None
                if "etok_ratio" in batch and "ktoe_ratio" in batch:
                    cs_ratios = (batch["etok_ratio"], batch["ktoe_ratio"])
                
                # Compute loss
                if loss_type == "custom":
                    loss, loss_components = loss_fn(embeddings_dict)
                else:
                    loss, loss_components = loss_fn(embeddings_dict, cs_ratios)
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
                
                # Track loss components
                epoch_total_loss += loss_components["total_loss"]
                for k in loss_components:
                    if k in epoch_loss_components:
                        epoch_loss_components[k] += loss_components[k]
            
            # Average loss over batches
            num_batches = len(train_dataloader)
            epoch_total_loss /= num_batches
            for k in epoch_loss_components:
                epoch_loss_components[k] /= num_batches
            
            # Track loss history
            self.loss_history["epoch"].append(epoch + 1)
            self.loss_history["total_loss"].append(epoch_total_loss)
            for k in epoch_loss_components:
                if k in self.loss_history:
                    self.loss_history[k].append(epoch_loss_components[k])
            
            # Print progress
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_total_loss:.4f}")
            for k, v in epoch_loss_components.items():
                if v > 0:  # Only print non-zero components
                    print(f"  {k}: {v:.4f}")
            
            # Save best model
            if epoch_total_loss < best_loss:
                best_loss = epoch_total_loss
                self.save_model(f"{self.results_dir}/best_model_{loss_type}")
            
            # Save checkpoint model
            if (epoch + 1) % 5 == 0 or epoch + 1 == num_epochs:
                self.save_model(f"{self.results_dir}/model_{loss_type}_epoch_{epoch+1}")
        
        # Plot loss history
        self.plot_loss_history(loss_type)
        
        return self.loss_history

    def save_model(self, path):
        """Save model state dictionary"""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """Load model state dictionary"""
        self.model.load_state_dict(torch.load(path))
        return self

    def compute_embeddings(self, data_loader, model):
        """Compute embeddings using the specified model"""
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
                # Move all inputs to device
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
        """Compute embeddings for original and fine-tuned models"""
        original_embeddings = self.compute_embeddings(data_loader, self.original_model)
        finetuned_embeddings = self.compute_embeddings(data_loader, self.model)
        
        return original_embeddings, finetuned_embeddings
    
    def plot_loss_history(self, loss_type):
        """Plot loss components during training"""
        try:
            print(f"Plotting loss history for {loss_type} loss...")
            
            # Verify results directory exists and is writable
            if not os.path.exists(self.results_dir):
                print(f"Creating results directory: {self.results_dir}")
                os.makedirs(self.results_dir, exist_ok=True)
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Verify loss history data
            if not self.loss_history["epoch"]:
                print("Warning: No epochs in loss history. Nothing to plot.")
                return
                
            print(f"Plotting data for {len(self.loss_history['epoch'])} epochs")
            
            # Plot total loss
            plt.plot(self.loss_history["epoch"], self.loss_history["total_loss"], 
                    'o-', linewidth=2, label='Total Loss', color='navy')
            
            # Define colors for different loss components
            colors = {
                "contrastive_loss": "forestgreen",
                "cs_reg_loss": "darkorange",
                "cohesion_loss": "darkred",
                "separation_loss": "purple"
            }
            
            # Plot loss components that have values
            for component in ["contrastive_loss", "cs_reg_loss", "cohesion_loss", "separation_loss"]:
                if component in self.loss_history and any(self.loss_history[component]):
                    print(f"Plotting {component} component")
                    plt.plot(self.loss_history["epoch"], self.loss_history[component], 
                            'o-', linewidth=2, 
                            label=component.replace('_', ' ').title(),
                            color=colors.get(component, None))
                else:
                    print(f"Skipping {component} component (no data)")
            
            plt.title(f'Training Loss ({loss_type})', fontsize=15)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=10)
            plt.tight_layout()
            
            # Save figure
            save_path = f"{self.results_dir}/loss_history_{loss_type}.png"
            print(f"Saving loss history plot to {save_path}")
            
            try:
                plt.savefig(save_path)
                print(f"Successfully saved loss history plot to {save_path}")
            except Exception as e:
                print(f"Error saving loss history plot to {save_path}: {e}")
                
                # Try alternate location as fallback
                alt_path = f"./loss_history_{loss_type}.png"
                print(f"Trying to save to alternate location: {alt_path}")
                try:
                    plt.savefig(alt_path)
                    print(f"Successfully saved loss history plot to {alt_path}")
                except Exception as e2:
                    print(f"Error saving to alternate location: {e2}")
            
            # Save loss data as CSV for backup
            try:
                csv_path = f"{self.results_dir}/loss_history_{loss_type}.csv"
                print(f"Saving loss data to CSV: {csv_path}")
                
                with open(csv_path, 'w') as f:
                    # Write header
                    header = ['epoch'] + [comp for comp in self.loss_history.keys() if comp != 'epoch']
                    f.write(','.join(header) + '\n')
                    
                    # Write data rows
                    for i, epoch in enumerate(self.loss_history['epoch']):
                        row = [str(epoch)]
                        for comp in header[1:]:
                            if i < len(self.loss_history[comp]):
                                row.append(f"{self.loss_history[comp][i]:.6f}")
                            else:
                                row.append("N/A")
                        f.write(','.join(row) + '\n')
                        
                print(f"Successfully saved loss data to {csv_path}")
            except Exception as e:
                print(f"Error saving loss data to CSV: {e}")
            
            plt.close()
            print("Loss history plotting complete")
            
        except Exception as e:
            print(f"Error in plot_loss_history: {e}")
            import traceback
            traceback.print_exc()

    # def plot_loss_history(self, loss_type):
    #     """Plot loss components during training"""
    #     plt.figure(figsize=(12, 6))
        
    #     # Plot total loss
    #     plt.plot(self.loss_history["epoch"], self.loss_history["total_loss"], 
    #              'o-', linewidth=2, label='Total Loss')
        
    #     # Plot loss components that have values
    #     for component in ["contrastive_loss", "cs_reg_loss", "cohesion_loss", "separation_loss"]:
    #         if component in self.loss_history and any(self.loss_history[component]):
    #             plt.plot(self.loss_history["epoch"], self.loss_history[component], 
    #                     'o-', linewidth=2, label=component.replace('_', ' ').title())
        
    #     plt.title(f'Training Loss ({loss_type})', fontsize=15)
    #     plt.xlabel('Epoch', fontsize=12)
    #     plt.ylabel('Loss', fontsize=12)
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.legend(fontsize=10)
    #     plt.tight_layout()
        
    #     # Save figure
    #     plt.savefig(f"{self.results_dir}/loss_history_{loss_type}.png")
    #     plt.close()

    def visualize_embeddings(self, embeddings, title="Embeddings Visualization", method="pca"):
        """
        Visualize embeddings using dimensionality reduction
        
        Args:
            embeddings: Dictionary of embeddings
            title: Title for the plot
            method: 'pca' or 'tsne'
        """
        try:
            print(f"Visualizing embeddings using {method}...")
            
            # Stack all embeddings
            stacked_embs = np.vstack([embeddings[k] for k in embeddings])
            print(f"Stacked embeddings shape: {stacked_embs.shape}")
            
            # Apply dimensionality reduction
            if method == "pca":
                print("Applying PCA...")
                reducer = PCA(n_components=2)
                reduced_embs = reducer.fit_transform(stacked_embs)
                explained_var = reducer.explained_variance_ratio_
                method_name = "PCA"
                subtitle = f"(Explained variance: {explained_var[0]:.2%}, {explained_var[1]:.2%})"
                print(f"PCA complete. Explained variance: {explained_var[0]:.2%}, {explained_var[1]:.2%}")
            elif method == "tsne":
                print("Applying t-SNE (this may take a while)...")
                reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
                reduced_embs = reducer.fit_transform(stacked_embs)
                method_name = "t-SNE"
                subtitle = ""
                print("t-SNE complete.")
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
            
            print(f"Plot created for {method} visualization.")
            return plt
        
        except Exception as e:
            print(f"Error in visualize_embeddings ({method}): {e}")
            import traceback
            traceback.print_exc()
            # Return a default figure to prevent further errors
            return plt.figure()

    def visualize_with_pca(self, original_embeddings, finetuned_embeddings, loss_type=""):
        """
        Compare original and fine-tuned embeddings using PCA and t-SNE
        """
        try:
            # Verify results directory exists and is writable
            if not os.path.exists(self.results_dir):
                print(f"Creating results directory: {self.results_dir}")
                os.makedirs(self.results_dir, exist_ok=True)
            
            # Test write permissions
            test_file = os.path.join(self.results_dir, "test_write.txt")
            try:
                with open(test_file, 'w') as f:
                    f.write("Test write")
                os.remove(test_file)
                print(f"Directory {self.results_dir} is writable")
            except Exception as e:
                print(f"Warning: Cannot write to directory {self.results_dir}: {e}")
                print(f"Trying to use current directory instead")
                self.results_dir = "."
            
            # Print dimensions of embeddings for debugging
            for key in original_embeddings:
                print(f"Original {key} embeddings shape: {original_embeddings[key].shape}")
            for key in finetuned_embeddings:
                print(f"Finetuned {key} embeddings shape: {finetuned_embeddings[key].shape}")
            
            # Create PCA visualization
            print(f"\nGenerating PCA plot for original embeddings...")
            fig_pca_orig = self.visualize_embeddings(
                original_embeddings, 
                title="Original Model Embeddings (Before Fine-tuning)",
                method="pca"
            )
            save_path = f"{self.results_dir}/pca_original{loss_type}.png"
            print(f"Saving PCA plot to {save_path}")
            try:
                fig_pca_orig.savefig(save_path)
                print(f"Successfully saved PCA plot to {save_path}")
            except Exception as e:
                print(f"Error saving PCA plot to {save_path}: {e}")
            
            print(f"\nGenerating PCA plot for finetuned embeddings...")
            fig_pca_ft = self.visualize_embeddings(
                finetuned_embeddings, 
                title="Fine-tuned Model Embeddings",
                method="pca"
            )
            save_path = f"{self.results_dir}/pca_finetuned{loss_type}.png"
            print(f"Saving PCA plot to {save_path}")
            try:
                fig_pca_ft.savefig(save_path)
                print(f"Successfully saved PCA plot to {save_path}")
            except Exception as e:
                print(f"Error saving PCA plot to {save_path}: {e}")
            
            # Create t-SNE visualization
            print(f"\nGenerating t-SNE plot for original embeddings...")
            fig_tsne_orig = self.visualize_embeddings(
                original_embeddings, 
                title="Original Model Embeddings (Before Fine-tuning)",
                method="tsne"
            )
            save_path = f"{self.results_dir}/tsne_original{loss_type}.png"
            print(f"Saving t-SNE plot to {save_path}")
            try:
                fig_tsne_orig.savefig(save_path)
                print(f"Successfully saved t-SNE plot to {save_path}")
            except Exception as e:
                print(f"Error saving t-SNE plot to {save_path}: {e}")
            
            print(f"\nGenerating t-SNE plot for finetuned embeddings...")
            fig_tsne_ft = self.visualize_embeddings(
                finetuned_embeddings, 
                title="Fine-tuned Model Embeddings",
                method="tsne"
            )
            save_path = f"{self.results_dir}/tsne_finetuned{loss_type}.png"
            print(f"Saving t-SNE plot to {save_path}")
            try:
                fig_tsne_ft.savefig(save_path)
                print(f"Successfully saved t-SNE plot to {save_path}")
            except Exception as e:
                print(f"Error saving t-SNE plot to {save_path}: {e}")
            
            # Create side-by-side comparison
            print("\nGenerating side-by-side comparison plots...")
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
            
            try:
                # PCA Original
                print("Processing PCA for original model...")
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
                print("Processing PCA for fine-tuned model...")
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
                print("Processing t-SNE for original model...")
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
                print("Processing t-SNE for fine-tuned model...")
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
                
                plt.suptitle(f"Comparing Embeddings Before and After Fine-tuning ({loss_type})", fontsize=18)
                plt.tight_layout()
                plt.subplots_adjust(top=0.94)
                
                save_path = f"{self.results_dir}/embedding_comparison_{loss_type}.png"
                print(f"Saving comparison plot to {save_path}")
                plt.savefig(save_path, dpi=300)
                print(f"Successfully saved comparison plot to {save_path}")
                plt.close()
            
            except Exception as e:
                print(f"Error creating comparison plots: {e}")
                import traceback
                traceback.print_exc()
            
            # Calculate and report average distances
            print("\nCalculating average L2 distances...")
            print("Before fine-tuning:")
            before_distances = {}
            for lang1 in original_embeddings:
                for lang2 in original_embeddings:
                    if lang1 < lang2:  # To avoid duplicates
                        try:
                            avg_dist = np.mean(np.linalg.norm(
                                original_embeddings[lang1] - original_embeddings[lang2], axis=1
                            ))
                            before_distances[f"{lang1}-{lang2}"] = avg_dist
                            print(f"  {lang1} to {lang2}: {avg_dist:.4f}")
                        except Exception as e:
                            print(f"  Error calculating distance from {lang1} to {lang2}: {e}")
            
            print("\nAfter fine-tuning:")
            after_distances = {}
            for lang1 in finetuned_embeddings:
                for lang2 in finetuned_embeddings:
                    if lang1 < lang2:  # To avoid duplicates
                        try:
                            avg_dist = np.mean(np.linalg.norm(
                                finetuned_embeddings[lang1] - finetuned_embeddings[lang2], axis=1
                            ))
                            after_distances[f"{lang1}-{lang2}"] = avg_dist
                            print(f"  {lang1} to {lang2}: {avg_dist:.4f}")
                        except Exception as e:
                            print(f"  Error calculating distance from {lang1} to {lang2}: {e}")
            
            # Save distances to CSV
            try:
                print("\nCreating distance comparison plot...")
                distances_before = []
                distances_after = []
                pairs = []
                
                for lang1 in original_embeddings:
                    for lang2 in original_embeddings:
                        if lang1 < lang2:
                            pair = f"{lang1}-{lang2}"
                            pairs.append(pair)
                            distances_before.append(before_distances.get(pair, 0))
                            distances_after.append(after_distances.get(pair, 0))
                
                # Create distance comparison plot
                plt.figure(figsize=(12, 6))
                x = np.arange(len(pairs))
                width = 0.35
                
                plt.bar(x - width/2, distances_before, width, label='Before Fine-tuning')
                plt.bar(x + width/2, distances_after, width, label='After Fine-tuning')
                
                plt.xlabel('Language Pairs')
                plt.ylabel('Average L2 Distance')
                plt.title(f'Distance Between Embeddings ({loss_type})')
                plt.xticks(x, pairs, rotation=45)
                plt.legend()
                plt.tight_layout()
                
                save_path = f"{self.results_dir}/distance_comparison_{loss_type}.png"
                print(f"Saving distance comparison plot to {save_path}")
                plt.savefig(save_path)
                print(f"Successfully saved distance comparison plot to {save_path}")
                plt.close()
                
                # Save distances to text file as backup
                with open(f"{self.results_dir}/distances_{loss_type}.txt", 'w') as f:
                    f.write("Pair,Before,After,Change\n")
                    for i, pair in enumerate(pairs):
                        before = distances_before[i]
                        after = distances_after[i]
                        change = after - before
                        f.write(f"{pair},{before:.4f},{after:.4f},{change:.4f}\n")
                        
                print(f"Saved distance data to {self.results_dir}/distances_{loss_type}.txt")
                
            except Exception as e:
                print(f"Error creating distance comparison: {e}")
                import traceback
                traceback.print_exc()
            
            # Return embedding statistics for further analysis
            stats = {
                "pairs": pairs,
                "before": distances_before,
                "after": distances_after,
                "change": [(after - before) for before, after in zip(distances_before, distances_after)]
            }
            
            return stats
        
        except Exception as e:
            print(f"Error in visualize_with_pca: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    # def visualize_embeddings(self, embeddings, title="Embeddings Visualization", method="pca"):
    #     """
    #     Visualize embeddings using dimensionality reduction
        
    #     Args:
    #         embeddings: Dictionary of embeddings
    #         title: Title for the plot
    #         method: 'pca' or 'tsne'
    #     """
    #     # Stack all embeddings
    #     stacked_embs = np.vstack([embeddings[k] for k in embeddings])
        
    #     # Apply dimensionality reduction
    #     if method == "pca":
    #         reducer = PCA(n_components=2)
    #         reduced_embs = reducer.fit_transform(stacked_embs)
    #         explained_var = reducer.explained_variance_ratio_
    #         method_name = "PCA"
    #         subtitle = f"(Explained variance: {explained_var[0]:.2%}, {explained_var[1]:.2%})"
    #     elif method == "tsne":
    #         reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    #         reduced_embs = reducer.fit_transform(stacked_embs)
    #         method_name = "t-SNE"
    #         subtitle = ""
    #     else:
    #         raise ValueError(f"Unknown visualization method: {method}")
        
    #     # Split reduced embeddings back by language
    #     start_idx = 0
    #     reduced_by_lang = {}
    #     for lang, embs in embeddings.items():
    #         end_idx = start_idx + embs.shape[0]
    #         reduced_by_lang[lang] = reduced_embs[start_idx:end_idx]
    #         start_idx = end_idx
        
    #     # Plotting
    #     plt.figure(figsize=(10, 8))
        
    #     # Define colors and markers
    #     colors = {
    #         "english": "blue",
    #         "etok": "green",
    #         "ktoe": "red",
    #         "korean": "purple"
    #     }
        
    #     markers = {
    #         "english": "o",
    #         "etok": "s",
    #         "ktoe": "^",
    #         "korean": "D"
    #     }
        
    #     # Plot each language variant
    #     for lang, embs in reduced_by_lang.items():
    #         plt.scatter(
    #             embs[:, 0], embs[:, 1],
    #             color=colors[lang],
    #             marker=markers[lang],
    #             label=lang.capitalize(),
    #             alpha=0.7,
    #             s=70
    #         )
        
    #     plt.title(f"{title}\n{method_name} {subtitle}", fontsize=15)
    #     plt.xlabel(f"{method_name} Component 1", fontsize=12)
    #     plt.ylabel(f"{method_name} Component 2", fontsize=12)
    #     plt.grid(True, linestyle='--', alpha=0.3)
    #     plt.legend(fontsize=10)
    #     plt.tight_layout()
        
    #     return plt
    
    # def visualize_with_pca(self, original_embeddings, finetuned_embeddings, loss_type=""):
    #     """
    #     Compare original and fine-tuned embeddings using PCA and t-SNE
    #     """
    #     # Create PCA visualization
    #     fig_pca_orig = self.visualize_embeddings(
    #         original_embeddings, 
    #         title="Original Model Embeddings (Before Fine-tuning)",
    #         method="pca"
    #     )
    #     fig_pca_orig.savefig(f"{self.results_dir}/pca_original{loss_type}.png")
        
    #     fig_pca_ft = self.visualize_embeddings(
    #         finetuned_embeddings, 
    #         title="Fine-tuned Model Embeddings",
    #         method="pca"
    #     )
    #     fig_pca_ft.savefig(f"{self.results_dir}/pca_finetuned{loss_type}.png")
        
    #     # Create t-SNE visualization
    #     fig_tsne_orig = self.visualize_embeddings(
    #         original_embeddings, 
    #         title="Original Model Embeddings (Before Fine-tuning)",
    #         method="tsne"
    #     )
    #     fig_tsne_orig.savefig(f"{self.results_dir}/tsne_original{loss_type}.png")
        
    #     fig_tsne_ft = self.visualize_embeddings(
    #         finetuned_embeddings, 
    #         title="Fine-tuned Model Embeddings",
    #         method="tsne"
    #     )
    #     fig_tsne_ft.savefig(f"{self.results_dir}/tsne_finetuned{loss_type}.png")
        
    #     # Create side-by-side comparison
    #     fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
    #     # Define colors and markers
    #     colors = {
    #         "english": "blue",
    #         "etok": "green",
    #         "ktoe": "red",
    #         "korean": "purple"
    #     }
        
    #     markers = {
    #         "english": "o",
    #         "etok": "s",
    #         "ktoe": "^",
    #         "korean": "D"
    #     }
        
    #     # PCA Original
    #     for lang, embs in original_embeddings.items():
    #         pca = PCA(n_components=2)
    #         reduced = pca.fit_transform(embs)
    #         axes[0, 0].scatter(
    #             reduced[:, 0], reduced[:, 1],
    #             color=colors[lang],
    #             marker=markers[lang],
    #             label=lang.capitalize(),
    #             alpha=0.7,
    #             s=70
    #         )
    #     axes[0, 0].set_title("Original Model - PCA", fontsize=14)
    #     axes[0, 0].grid(True, linestyle='--', alpha=0.3)
    #     axes[0, 0].legend(fontsize=10)
        
    #     # PCA Fine-tuned
    #     for lang, embs in finetuned_embeddings.items():
    #         pca = PCA(n_components=2)
    #         reduced = pca.fit_transform(embs)
    #         axes[0, 1].scatter(
    #             reduced[:, 0], reduced[:, 1],
    #             color=colors[lang],
    #             marker=markers[lang],
    #             label=lang.capitalize(),
    #             alpha=0.7,
    #             s=70
    #         )
    #     axes[0, 1].set_title("Fine-tuned Model - PCA", fontsize=14)
    #     axes[0, 1].grid(True, linestyle='--', alpha=0.3)
    #     axes[0, 1].legend(fontsize=10)
        
    #     # t-SNE Original
    #     all_embs_orig = np.vstack([original_embeddings[k] for k in original_embeddings])
    #     tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    #     reduced_orig = tsne.fit_transform(all_embs_orig)
        
    #     start_idx = 0
    #     for lang, embs in original_embeddings.items():
    #         end_idx = start_idx + embs.shape[0]
    #         axes[1, 0].scatter(
    #             reduced_orig[start_idx:end_idx, 0], 
    #             reduced_orig[start_idx:end_idx, 1],
    #             color=colors[lang],
    #             marker=markers[lang],
    #             label=lang.capitalize(),
    #             alpha=0.7,
    #             s=70
    #         )
    #         start_idx = end_idx
    #     axes[1, 0].set_title("Original Model - t-SNE", fontsize=14)
    #     axes[1, 0].grid(True, linestyle='--', alpha=0.3)
    #     axes[1, 0].legend(fontsize=10)
        
    #     # t-SNE Fine-tuned
    #     all_embs_ft = np.vstack([finetuned_embeddings[k] for k in finetuned_embeddings])
    #     tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    #     reduced_ft = tsne.fit_transform(all_embs_ft)
        
    #     start_idx = 0
    #     for lang, embs in finetuned_embeddings.items():
    #         end_idx = start_idx + embs.shape[0]
    #         axes[1, 1].scatter(
    #             reduced_ft[start_idx:end_idx, 0], 
    #             reduced_ft[start_idx:end_idx, 1],
    #             color=colors[lang],
    #             marker=markers[lang],
    #             label=lang.capitalize(),
    #             alpha=0.7,
    #             s=70
    #         )
    #         start_idx = end_idx
    #     axes[1, 1].set_title("Fine-tuned Model - t-SNE", fontsize=14)
    #     axes[1, 1].grid(True, linestyle='--', alpha=0.3)
    #     axes[1, 1].legend(fontsize=10)
        
    #     plt.suptitle(f"Comparing Embeddings Before and After Fine-tuning ({loss_type})", fontsize=18)
    #     plt.tight_layout()
    #     plt.subplots_adjust(top=0.94)
    #     plt.savefig(f"{self.results_dir}/embedding_comparison_{loss_type}.png", dpi=300)
    #     plt.close()
        
    #     # Calculate and report average distances
    #     print("\nAverage L2 distances:")
    #     print("Before fine-tuning:")
    #     for lang1 in original_embeddings:
    #         for lang2 in original_embeddings:
    #             if lang1 < lang2:  # To avoid duplicates
    #                 avg_dist = np.mean(np.linalg.norm(
    #                     original_embeddings[lang1] - original_embeddings[lang2], axis=1
    #                 ))
    #                 print(f"  {lang1} to {lang2}: {avg_dist:.4f}")
        
    #     print("\nAfter fine-tuning:")
    #     for lang1 in finetuned_embeddings:
    #         for lang2 in finetuned_embeddings:
    #             if lang1 < lang2:  # To avoid duplicates
    #                 avg_dist = np.mean(np.linalg.norm(
    #                     finetuned_embeddings[lang1] - finetuned_embeddings[lang2], axis=1
    #                 ))
    #                 print(f"  {lang1} to {lang2}: {avg_dist:.4f}")
        
    #     # Save distances to CSV
    #     distances_before = []
    #     distances_after = []
    #     pairs = []
        
    #     for lang1 in original_embeddings:
    #         for lang2 in original_embeddings:
    #             if lang1 < lang2:
    #                 pairs.append(f"{lang1}-{lang2}")
    #                 distances_before.append(np.mean(np.linalg.norm(
    #                     original_embeddings[lang1] - original_embeddings[lang2], axis=1
    #                 )))
    #                 distances_after.append(np.mean(np.linalg.norm(
    #                     finetuned_embeddings[lang1] - finetuned_embeddings[lang2], axis=1
    #                 )))
        
    #     # Create distance comparison plot
    #     plt.figure(figsize=(12, 6))
    #     x = np.arange(len(pairs))
    #     width = 0.35
        
    #     plt.bar(x - width/2, distances_before, width, label='Before Fine-tuning')
    #     plt.bar(x + width/2, distances_after, width, label='After Fine-tuning')
        
    #     plt.xlabel('Language Pairs')
    #     plt.ylabel('Average L2 Distance')
    #     plt.title(f'Distance Between Embeddings ({loss_type})')
    #     plt.xticks(x, pairs, rotation=45)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(f"{self.results_dir}/distance_comparison_{loss_type}.png")
    #     plt.close()
        
    #     # Return embedding statistics for further analysis
    #     stats = {
    #         "pairs": pairs,
    #         "before": distances_before,
    #         "after": distances_after,
    #         "change": [(after - before) for before, after in zip(distances_before, distances_after)]
    #     }
        
    #     return stats

# Main function to execute the pipeline
def main():
    # Load data
    print("Loading data...")
    with open("data/code-switch.json", "r", encoding="UTF-8") as f:
        data_list = json.load(f)
    
    # Initialize finetuner
    print("Initializing finetuner...")
    finetuner = EmbeddingFinetuner()
    
    # Create dataset
    print("Creating dataset...")
    dataset = MultilingualDataset(data_list=data_list, tokenizer=finetuner.tokenizer)
    
    # Create dataloader
    print("Creating dataloader...")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # For comparison, fine-tune with different loss functions
    loss_types = ["refined", "custom", "codeswitch"]
    loss_type = loss_types[0]
    
    finetuner = EmbeddingFinetuner()
    loss_history = finetuner.finetune(dataloader, loss_type=loss_type, num_epochs=10, lr=1e-5)
        
    # Save the fine-tuned model
    print(f"Saving fine-tuned model with {loss_type} loss...")
    finetuner.save_model(f"finetuned_multilingual_model_{loss_type}")
    # Evaluate embeddings
    print(f"Evaluating embeddings with {loss_type} loss...")
    original_embeddings, finetuned_embeddings = finetuner.evaluate_embeddings(dataloader)
    
    # Visualize results
    print(f"Visualizing results with {loss_type} loss...")
    embedding_stats = finetuner.visualize_with_pca(original_embeddings, finetuned_embeddings, loss_type)
    
    print(f"\nTraining with {loss_type} loss completed!\n")
    # for loss_type in loss_types:
    #     print(f"\n\n========== Training with {loss_type} loss ==========")
        
    #     # Reset model to original weights
    #     finetuner = EmbeddingFinetuner()
        
    #     # Fine-tune the model
    #     loss_history = finetuner.finetune(dataloader, loss_type=loss_type, num_epochs=10, lr=1e-5)
        
    #     # Save the fine-tuned model
    #     print(f"Saving fine-tuned model with {loss_type} loss...")
    #     finetuner.save_model(f"finetuned_multilingual_model_{loss_type}")
        
    #     # Evaluate embeddings
    #     print(f"Evaluating embeddings with {loss_type} loss...")
    #     original_embeddings, finetuned_embeddings = finetuner.evaluate_embeddings(dataloader)
        
    #     # Visualize results
    #     print(f"Visualizing results with {loss_type} loss...")
    #     embedding_stats = finetuner.visualize_with_pca(original_embeddings, finetuned_embeddings, loss_type)
        
    #     print(f"\nTraining with {loss_type} loss completed!\n")
    
    print("\nAll experiments completed successfully!")

if __name__ == "__main__":
    main()