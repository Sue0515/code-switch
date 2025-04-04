import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import json
import datetime
import random

# Define the dataset class for handling code-switch.json
class MultilingualDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=128):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

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

        return {
            "english_input_ids": english["input_ids"].squeeze(0),
            "english_attention_mask": english["attention_mask"].squeeze(0),
            "etok_input_ids": etok["input_ids"].squeeze(0),
            "etok_attention_mask": etok["attention_mask"].squeeze(0),
            "ktoe_input_ids": ktoe["input_ids"].squeeze(0),
            "ktoe_attention_mask": ktoe["attention_mask"].squeeze(0),
            "korean_input_ids": korean["input_ids"].squeeze(0),
            "korean_attention_mask": korean["attention_mask"].squeeze(0)
        }

class RefinedCodeSwitchLoss(nn.Module):
    def __init__(self, temperature=0.1, lambda_cs_reg=0.5):
        super(RefinedCodeSwitchLoss, self).__init__()
        self.temperature = temperature
        self.lambda_cs_reg = lambda_cs_reg
        
    def forward(self, embeddings_dict, cs_ratios):
        """
        Args:
            embeddings_dict: Dictionary with embeddings
            cs_ratios: Tensor of shape (batch_size,) with values between 0-1
                       indicating how much English vs Korean in each example
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
        for i in range(batch_size):
            ratio = cs_ratios[i]
            
            # EtoK should be ratio% similar to English, (1-ratio)% similar to Korean
            etok_target = ratio * english_emb[i] + (1 - ratio) * korean_emb[i]
            etok_reg = torch.norm(etok_emb[i] - etok_target)
            
            # KtoE should be (1-ratio)% similar to English, ratio% similar to Korean
            ktoe_target = (1 - ratio) * english_emb[i] + ratio * korean_emb[i] 
            ktoe_reg = torch.norm(ktoe_emb[i] - ktoe_target)
            
            cs_reg_loss += etok_reg + ktoe_reg
        
        # Normalize losses by batch size
        contrastive_loss = contrastive_loss / (2 * batch_size)  # 2 anchors per example
        cs_reg_loss = cs_reg_loss / batch_size
        
        # Combine losses
        total_loss = contrastive_loss + self.lambda_cs_reg * cs_reg_loss
        
        # Return both total loss and components for tracking
        components = {
            "contrastive_loss": contrastive_loss.item(),
            "cs_reg_loss": cs_reg_loss.item(),
            "total_loss": total_loss.item()
        }
        
        return total_loss, components
    
    
class CodeSwitchLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(CodeSwitchLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, embeddings_dict, cs_ratios):
        """
        Args:
            embeddings_dict: Dictionary with embeddings
            cs_ratios: Tensor of shape (batch_size,) with values between 0-1
                       indicating how much English vs Korean in each example
        """
        # Normalize embeddings
        english_emb = nn.functional.normalize(embeddings_dict["english"], p=2, dim=1)
        etok_emb = nn.functional.normalize(embeddings_dict["etok"], p=2, dim=1)
        ktoe_emb = nn.functional.normalize(embeddings_dict["ktoe"], p=2, dim=1)
        korean_emb = nn.functional.normalize(embeddings_dict["korean"], p=2, dim=1)
        
        batch_size = english_emb.shape[0]
        
        # 1. Basic contrastive loss (similar to your current approach)
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
        for i in range(batch_size):
            ratio = cs_ratios[i]
            
            # EtoK should be ratio% similar to English, (1-ratio)% similar to Korean
            etok_target = ratio * english_emb[i] + (1 - ratio) * korean_emb[i]
            etok_reg = torch.norm(etok_emb[i] - etok_target)
            
            # KtoE should be (1-ratio)% similar to English, ratio% similar to Korean
            ktoe_target = (1 - ratio) * english_emb[i] + ratio * korean_emb[i] 
            ktoe_reg = torch.norm(ktoe_emb[i] - ktoe_target)
            
            cs_reg_loss += etok_reg + ktoe_reg
        
        # Combine losses
        total_loss = contrastive_loss + 0.5 * cs_reg_loss
        
        return total_loss / batch_size
    
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

        return total_loss


# Define the finetuning class
class EmbeddingFinetuner:
    def __init__(self, model_name="BAAI/bge-m3", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

        # Save the original model for comparison
        self.original_model = AutoModel.from_pretrained(model_name).to(device)
        self.original_model.eval()  # Set to evaluation mode
    

    def get_embedding(self, model, input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output

    def finetune(self, train_dataloader, num_epochs=5, lr=1e-5):
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        loss_fn = ImprovedCustomLoss()

        self.model.train()

        for epoch in range(num_epochs):
            total_loss = 0.0

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

                # Compute custom loss and optimize
                loss = loss_fn(embeddings_dict)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader):.4f}")

    def save_model(self, path="finetuned_model"):
        torch.save(self.model.state_dict(), path)


    def compute_embeddings(self, data_loader, model):
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
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
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
        original_embeddings = self.compute_embeddings(data_loader, self.original_model)
        finetuned_embeddings = self.compute_embeddings(data_loader, self.model)
        
        return original_embeddings, finetuned_embeddings
    
    def visualize_with_pca(self, original_embeddings, finetuned_embeddings):
        # Stack all embeddings for fitting PCA
        original_stack = np.vstack([original_embeddings[k] for k in original_embeddings])
        finetuned_stack = np.vstack([finetuned_embeddings[k] for k in finetuned_embeddings])
        
        # Fit PCA on combined data
        pca_original = PCA(n_components=2)
        pca_finetuned = PCA(n_components=2)
        
        # Transform each set of embeddings
        original_pca_result = {}
        finetuned_pca_result = {}
        
        # Fit PCA on all original embeddings
        pca_original.fit(original_stack)
        for key in original_embeddings:
            original_pca_result[key] = pca_original.transform(original_embeddings[key])
        
        # Fit PCA on all finetuned embeddings
        pca_finetuned.fit(finetuned_stack)
        for key in finetuned_embeddings:
            finetuned_pca_result[key] = pca_finetuned.transform(finetuned_embeddings[key])
        
        # Create figure for visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Define colors and markers for each type
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
        
        # Plot original embeddings
        for key in original_pca_result:
            ax1.scatter(
                original_pca_result[key][:, 0], 
                original_pca_result[key][:, 1],
                color=colors[key],
                marker=markers[key],
                label=key,
                alpha=0.7
            )
        
        ax1.set_title("Original Model Embeddings (Before Fine-tuning)")
        ax1.legend()
        ax1.grid(True)
        
        # Plot finetuned embeddings
        for key in finetuned_pca_result:
            ax2.scatter(
                finetuned_pca_result[key][:, 0], 
                finetuned_pca_result[key][:, 1],
                color=colors[key],
                marker=markers[key],
                label=key,
                alpha=0.7
            )
        
        ax2.set_title("Fine-tuned Model Embeddings")
        ax2.legend()
        ax2.grid(True)
        
        plt.suptitle("Comparing Embeddings Before and After Fine-tuning")
        plt.tight_layout()
        plt.savefig("embedding_comparison.png")
        plt.show()
        
        # Calculate and report average distances
        print("\nAverage L2 distances from English embeddings:")
        print("Before fine-tuning:")
        for key in ["etok", "ktoe", "korean"]:
            avg_dist = np.mean(np.linalg.norm(
                original_embeddings["english"] - original_embeddings[key], axis=1
            ))
            print(f"  English to {key}: {avg_dist:.4f}")
        
        print("\nAfter fine-tuning:")
        for key in ["etok", "ktoe", "korean"]:
            avg_dist = np.mean(np.linalg.norm(
                finetuned_embeddings["english"] - finetuned_embeddings[key], axis=1
            ))
            print(f"  English to {key}: {avg_dist:.4f}")




# Main function to execute the pipeline
def main():
    with open("code-switch.json", "r",encoding="UTF-8") as f:
        data_list = json.load(f)

    finetuner = EmbeddingFinetuner()
    dataset = MultilingualDataset(data_list=data_list, tokenizer=finetuner.tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Fine-tune the model
    finetuner.finetune(dataloader, num_epochs=10, lr=1e-5)

    # Save the fine-tuned model
    finetuner.save_model("finetuned_multilingual_model")

    # Evaluate embeddings
    original_embeddings, finetuned_embeddings = finetuner.evaluate_embeddings(dataloader)
    
    # Visualize results
    finetuner.visualize_with_pca(original_embeddings, finetuned_embeddings)

if __name__ == "__main__":
    main()
