import torch
import torch.nn as nn
import torch.nn.functional as F

class RefinedCodeSwitchLoss(nn.Module):

    def __init__(self, temperature=0.1, lambda_cs_reg=0.5):
        """
        Args:
            temperature: Temperature for contrastive loss scaling
            lambda_cs_reg: Weight for code-switching regularization loss
        """
        super(RefinedCodeSwitchLoss, self).__init__()
        self.temperature = temperature
        self.lambda_cs_reg = lambda_cs_reg
        
    def forward(self, embeddings_dict, cs_ratios=None):
        # Normalize embeddings
        english_emb = F.normalize(embeddings_dict["english"], p=2, dim=1)
        etok_emb = F.normalize(embeddings_dict["etok"], p=2, dim=1)
        ktoe_emb = F.normalize(embeddings_dict["ktoe"], p=2, dim=1)
        korean_emb = F.normalize(embeddings_dict["korean"], p=2, dim=1)
        
        batch_size = english_emb.shape[0]
        
        # 1. Contrastive loss using English and Korean as anchors
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