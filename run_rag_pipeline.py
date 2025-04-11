import os
import argparse
import json
import logging
from typing import Dict, Any

from rag.config import RAGConfig
from rag.document import Document
from rag.embedding import EmbeddingModel
from rag.retriever import WikipediaRetriever
from rag.system import RAGSystem
from rag.evaluation import RAGEvaluator
from utils.logger import setup_logger

# Set up logging
logger = setup_logger(name="rag_pipeline", level=logging.INFO)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run the multilingual code-switched RAG pipeline"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--query_file", 
        type=str, 
        help="Path to query JSON file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        help="Output directory for results"
    )
    parser.add_argument(
        "--base_model", 
        type=str, 
        help="Base model name"
    )
    parser.add_argument(
        "--finetuned_model", 
        type=str, 
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--corpus_path", 
        type=str, 
        help="Path to existing corpus (to skip corpus building)"
    )
    parser.add_argument(
        "--demo_query", 
        type=str, 
        help="Single query to process (demo mode)"
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Skip evaluation and just build the systems"
    )
    parser.add_argument(
        "--no_visualization",
        action="store_true",
        help="Disable visualization generation"
    )
    
    return parser.parse_args()

def build_corpus(config: RAGConfig) -> list:
    """
    Build or load the document corpus
    
    Args:
        config: RAG configuration
        
    Returns:
        List of Document objects
    """
    # Check if we should load existing corpus
    if config.corpus_path and os.path.exists(config.corpus_path):
        logger.info(f"Loading corpus from {config.corpus_path}")
        corpus = WikipediaRetriever.load_corpus(config.corpus_path)
        logger.info(f"Loaded {len(corpus)} documents from disk")
        return corpus
    
    # Otherwise, build corpus from Wikipedia
    if config.use_wikipedia:
        # Load queries
        with open(config.query_file, "r", encoding="utf-8") as f:
            queries = json.load(f)
        
        # Create Wikipedia retriever
        wiki_retriever = WikipediaRetriever(langs=config.wikipedia_langs)
        
        # Build corpus
        corpus = wiki_retriever.build_corpus(
            queries=queries,
            limit_per_query=config.documents_per_query
        )
        
        # Save corpus if requested
        if config.save_corpus:
            corpus_path = os.path.join(config.output_dir, "corpus.json")
            wiki_retriever.save_corpus(corpus, corpus_path)
            logger.info(f"Saved corpus to {corpus_path}")
        
        return corpus
    
    logger.warning("No corpus source specified or found")
    return []

def setup_rag_systems(config: RAGConfig) -> tuple:
    """
    Set up base and fine-tuned RAG systems
    
    Args:
        config: RAG configuration
        
    Returns:
        Tuple of (base_system, finetuned_system)
    """
    # Initialize embedding models
    logger.info("Initializing embedding models")
    
    # Base model
    base_model = EmbeddingModel(
        model_name_or_path=config.base_model_name,
        device=config.device
    )
    
    # Fine-tuned model
    finetuned_model = EmbeddingModel(
        model_name_or_path=config.base_model_name,
        finetuned_model_path=config.finetuned_model_path,
        finetuned_weights_path=config.finetuned_weights_path,
        device=config.device
    )
    
    # Initialize RAG systems
    logger.info("Initializing RAG systems")
    base_rag = RAGSystem(
        embedding_model=base_model,
        name="base",
        metric_type=config.metric_type
    )
    
    finetuned_rag = RAGSystem(
        embedding_model=finetuned_model,
        name="finetuned",
        metric_type=config.metric_type
    )
    
    # Build or load corpus
    corpus = build_corpus(config)
    
    # Add documents to RAG systems
    if corpus:
        logger.info("Adding documents to RAG systems")
        base_rag.add_documents(corpus)
        finetuned_rag.add_documents(corpus)
        
        # Save RAG systems
        logger.info("Saving RAG systems")
        base_rag.save(config.output_dir)
        finetuned_rag.save(config.output_dir)
    else:
        logger.warning("No corpus available, systems will not be populated with documents")
    
    return base_rag, finetuned_rag

def evaluate_systems(base_rag: RAGSystem, finetuned_rag: RAGSystem, config: RAGConfig) -> str:
    """
    Evaluate RAG systems on queries
    
    Args:
        base_rag: Base RAG system
        finetuned_rag: Fine-tuned RAG system
        config: RAG configuration
        
    Returns:
        Path to evaluation report
    """
    # Load queries
    with open(config.query_file, "r", encoding="utf-8") as f:
        queries = json.load(f)
    
    # Initialize evaluator
    evaluator = RAGEvaluator(
        base_system=base_rag,
        finetuned_system=finetuned_rag,
        output_dir=config.output_dir,
        visualization=not config.no_visualization
    )
    
    # Evaluate queries
    evaluator.evaluate_queries(queries, top_k=config.top_k)
    
    # Generate and save report
    report_path = evaluator.generate_report()
    
    # Export results to CSV for further analysis
    csv_path = evaluator.export_results_to_csv()
    
    return report_path

def process_demo_query(query: str, base_rag: RAGSystem, finetuned_rag: RAGSystem) -> None:
    logger.info(f"Processing demo query: {query}")
    
    # Detect language
    query_language = base_rag.embedding_model.detect_language(query)
    logger.info(f"Detected language: {query_language}")
    
    # Get results from both systems
    base_results = base_rag.retrieve(query, top_k=5)
    finetuned_results = finetuned_rag.retrieve(query, top_k=5)
    
    # Display results
    print("\n===== DEMO QUERY =====")
    print(f"Query: {query}")
    print(f"Detected language: {query_language}")
    
    print("\n===== BASE MODEL RESULTS =====")
    for i, res in enumerate(base_results, 1):
        print(f"{i}. [{res['language']}] {res['title']}")
        print(f"   Score: {res['score']:.4f}")
        print(f"   Content: {res['content'][:100]}...")
    
    print("\n===== FINETUNED MODEL RESULTS =====")
    for i, res in enumerate(finetuned_results, 1):
        print(f"{i}. [{res['language']}] {res['title']}")
        print(f"   Score: {res['score']:.4f}")
        print(f"   Content: {res['content'][:100]}...")

def main():
    """Main function to run the RAG pipeline"""
    args = parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        logger.info(f"Loading configuration from {args.config}")
        config = RAGConfig.from_json(args.config)
    else:
        logger.info("Using default configuration with command line overrides")
        config = RAGConfig()
    
    # Override from command line arguments
    if args.query_file:
        config.query_file = args.query_file
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.base_model:
        config.base_model_name = args.base_model
    if args.finetuned_model:
        config.finetuned_model_path = args.finetuned_model
    if args.corpus_path:
        config.corpus_path = args.corpus_path
    if args.no_visualization:
        config.visualization = False
    
    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config.output_dir, "config.json")
    config.save(config_path)
    
    # Set up RAG systems
    logger.info("Setting up RAG systems")
    base_rag, finetuned_rag = setup_rag_systems(config)
    
    # Process a single query in demo mode
    if args.demo_query:
        process_demo_query(args.demo_query, base_rag, finetuned_rag)
        return
    
    # Evaluate if not skipped
    if not args.skip_evaluation:
        logger.info("Evaluating RAG systems")
        report_path = evaluate_systems(base_rag, finetuned_rag, config)
        logger.info(f"Evaluation complete. Report available at {report_path}")
    else:
        logger.info("Evaluation skipped")
    
    logger.info("RAG pipeline execution complete")

if __name__ == "__main__":
    main()