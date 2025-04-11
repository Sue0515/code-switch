"""
Fine-tune multilingual embeddings for code-switched text.

This script fine-tunes a multilingual embedding model (e.g., BGE-M3)
to better handle code-switched text, particularly English-Korean language mixing.
"""
import os
import argparse
import logging
import json
import torch
from transformers import AutoTokenizer

from finetune.config import FinetuningConfig
from finetune.dataset import create_dataloader, MultilingualDataset
from finetune.model import EmbeddingModel
from finetune.trainer import EmbeddingTrainer
from finetune.visualization import EmbeddingVisualizer

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Fine-tune multilingual embeddings for code-switched text"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--data_file", 
        type=str, 
        help="Path to code-switch data JSON file"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="BAAI/bge-m3", 
        help="Base model name or path"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./results", 
        help="Output directory for results"
    )
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=10, 
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4, 
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-5, 
        help="Learning rate"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=128, 
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--no_visualize", 
        action="store_true", 
        help="Skip visualization of embeddings"
    )
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        logger.info(f"Loading configuration from {args.config}")
        config = FinetuningConfig.from_json(args.config)
    else:
        logger.info("Using default configuration with command line overrides")
        config = FinetuningConfig()
    
    # Override configuration from command line arguments
    if args.data_file:
        config.data_file = args.data_file
    if args.model_name:
        config.model_name = args.model_name
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.max_length:
        config.max_length = args.max_length
    if args.no_visualize:
        config.visualize_embeddings = False
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config.output_dir, "config.json")
    config.save(config_path)
    logger.info(f"Saved configuration to {config_path}")
    
    # Initialize model
    logger.info(f"Initializing model {config.model_name}")
    model = EmbeddingModel(
        model_name=config.model_name,
        device=config.device
    )
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Create dataloader
    logger.info(f"Loading data from {config.data_file}")
    with open(config.data_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    dataset = MultilingualDataset(
        data_list=data_list,
        tokenizer=tokenizer,
        max_length=config.max_length
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    # Create trainer
    trainer = EmbeddingTrainer(
        model=model,
        config=config
    )
    
    # Train model
    logger.info("Starting training")
    loss_history = trainer.train(dataloader)
    
    # Save loss history
    trainer.save_loss_history()
    
    # Evaluate embeddings
    if config.visualize_embeddings:
        logger.info("Evaluating embeddings")
        original_embeddings, finetuned_embeddings = model.evaluate_embeddings(dataloader)
        
        # Visualize results
        logger.info("Visualizing embeddings")
        visualizer = EmbeddingVisualizer(trainer.results_dir)
        
        # Plot loss history
        visualizer.plot_loss_history(loss_history)
        
        # Compare embeddings
        results = visualizer.compare_embeddings(
            original_embeddings,
            finetuned_embeddings
        )
        
        # Save results summary
        with open(os.path.join(trainer.results_dir, "visualization_results.json"), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            for key, value in results["distances"].items():
                if isinstance(value, list):
                    results["distances"][key] = [float(v) for v in value]
            
            json.dump(results, f, indent=2)
    
    logger.info(f"Fine-tuning complete. Results saved to {trainer.results_dir}")
    
    # Return path to fine-tuned model
    return os.path.join(trainer.results_dir, "final_model")

if __name__ == "__main__":
    main()