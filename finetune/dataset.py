import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class MultilingualDataset(Dataset):
    """Dataset for code-switched data with English, Korean, EtoK, and KtoE versions"""
    
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


def create_dataloader(config, tokenizer):

    # Load data
    with open(config.data_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    # Create dataset
    dataset = MultilingualDataset(
        data_list=data_list,
        tokenizer=tokenizer,
        max_length=config.max_length
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    return dataloader