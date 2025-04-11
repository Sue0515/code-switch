import os
import datetime
import torch
import torch.optim as optim
import logging
from tqdm import tqdm

from .loss import RefinedCodeSwitchLoss

logger = logging.getLogger(__name__)

class EmbeddingTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = os.path.join(config.output_dir, f"results_{timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.loss_fn = RefinedCodeSwitchLoss(
            temperature=config.temperature,
            lambda_cs_reg=config.lambda_cs_reg
        )
        
        self.optimizer = optim.AdamW(
            self.model.model.parameters(), 
            lr=config.learning_rate
        )
        
        # Initialize loss tracking
        self.loss_history = {
            "epoch": [],
            "total_loss": [],
            "contrastive_loss": [],
            "cs_reg_loss": []
        }
    
    def train(self, train_dataloader):
        self.model.save_original_model()
        self.model.model.train()
        
        # Track best model
        best_loss = float('inf')
        
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            epoch_total_loss = 0.0
            epoch_loss_components = {
                "contrastive_loss": 0.0,
                "cs_reg_loss": 0.0
            }
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            for batch in progress_bar:
                self.optimizer.zero_grad()
                
                # Move inputs to device
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                # Compute embeddings for all categories
                embeddings_dict = {
                    "english": self.model.get_embedding(
                        self.model.model, 
                        batch["english_input_ids"], 
                        batch["english_attention_mask"]
                    ),
                    "etok": self.model.get_embedding(
                        self.model.model, 
                        batch["etok_input_ids"], 
                        batch["etok_attention_mask"]
                    ),
                    "ktoe": self.model.get_embedding(
                        self.model.model, 
                        batch["ktoe_input_ids"], 
                        batch["ktoe_attention_mask"]
                    ),
                    "korean": self.model.get_embedding(
                        self.model.model, 
                        batch["korean_input_ids"], 
                        batch["korean_attention_mask"]
                    )
                }
                
                cs_ratios = None
                if "etok_ratio" in batch and "ktoe_ratio" in batch:
                    cs_ratios = (batch["etok_ratio"], batch["ktoe_ratio"])
                
                loss, loss_components = self.loss_fn(embeddings_dict, cs_ratios)
                
                loss.backward()
                self.optimizer.step()
                
                progress_bar.set_postfix({
                    "loss": f"{loss_components['total_loss']:.4f}"
                })
                
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
            
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}, Loss: {epoch_total_loss:.4f}")
            for k, v in epoch_loss_components.items():
                logger.info(f"  {k}: {v:.4f}")
            
            if epoch_total_loss < best_loss:
                best_loss = epoch_total_loss
                self.model.save_model(os.path.join(self.results_dir, "best_model"))
                logger.info(f"Saved best model with loss {best_loss:.4f}")
            
            if (epoch + 1) % 5 == 0 or epoch + 1 == self.config.num_epochs:
                self.model.save_model(os.path.join(self.results_dir, f"model_epoch_{epoch+1}"))
                logger.info(f"Saved checkpoint at epoch {epoch+1}")
        
        logger.info("Training complete")
        
        self.model.save_model(os.path.join(self.results_dir, "final_model"))
        
        return self.loss_history
    
    def save_loss_history(self):
        import pandas as pd
        
        # Convert to DataFrame
        df = pd.DataFrame(self.loss_history)
        
        # Save to CSV
        csv_path = os.path.join(self.results_dir, "loss_history.csv")
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Saved loss history to {csv_path}")
        
        return csv_path