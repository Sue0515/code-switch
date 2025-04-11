from transformers import AutoModel, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer

# 1. Initialize the base model and tokenizer
base_model = AutoModel.from_pretrained("BAAI/bge-m3")
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

# 2. Load the fine-tuned state dictionary
# Use the specific path to your model file, not just the directory
model_path = "./results_20250410_030859/best_model_refined"  # adjust to your specific model file
# Or: model_path = "./results_20250410_030859/model_refined_epoch_10"

# Load state dictionary and apply to the base model
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
base_model.load_state_dict(state_dict)

# 3. Create a SentenceTransformer model
from sentence_transformers.models import Transformer, Pooling
transformer = Transformer("BAAI/bge-m3")
transformer.auto_model = base_model  # Replace with fine-tuned model
pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")

st_model = SentenceTransformer(modules=[transformer, pooling])

# 4. Save the converted model
st_model.save("bge_m3_finetuned_suin")
print("Model saved to 'bge_m3_finetuned_suin'")


