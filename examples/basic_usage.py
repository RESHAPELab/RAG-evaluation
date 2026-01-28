"""
Example usage of the RAG Evaluation Framework

This script demonstrates how to use the framework to evaluate RAG models.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_evaluation import RAGEvaluator
from rag_evaluation.data_ingestion import JabrefLoader, DataTableLoader


def example_single_evaluation():
    """Example of evaluating a single RAG output."""
    print("=" * 60)
    print("Example 1: Single Evaluation")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Example data
    query = "What is machine learning?"
    context = (
        "Machine learning is a subset of artificial intelligence that enables "
        "systems to learn and improve from experience without being explicitly "
        "programmed. It focuses on developing computer programs that can access "
        "data and use it to learn for themselves."
    )
    answer = (
        "Machine learning is a type of artificial intelligence that allows "
        "computers to learn from data and improve their performance over time "
        "without explicit programming."
    )
    ground_truth = (
        "Machine learning is a subset of AI that enables systems to learn "
        "from experience and improve automatically."
    )
    
    # Evaluate
    results = evaluator.evaluate(
        query=query,
        context=context,
        answer=answer,
        ground_truth=ground_truth
    )
    
    # Display results
    print(f"\nQuery: {query}")
    print(f"\nAnswer: {answer}")
    print("\n" + "-" * 60)
    print("EVALUATION RESULTS:")
    print("-" * 60)
    
    for metric_name, result in results.items():
        print(f"\n{metric_name.upper()}:")
        print(f"  Score: {result['score']:.3f}")
        if 'details' in result:
            print(f"  Details: {result['details'].get('reasoning', '')}")
    
    print("\n")


def example_batch_evaluation():
    """Example of evaluating multiple RAG outputs."""
    print("=" * 60)
    print("Example 2: Batch Evaluation")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Example batch data
    queries = [
        "What is deep learning?",
        "What is natural language processing?",
        "What is computer vision?"
    ]
    
    contexts = [
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
        "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language.",
        "Computer vision is a field of AI that enables computers to understand and interpret visual information from images and videos."
    ]
    
    answers = [
        "Deep learning uses multi-layer neural networks to learn from data.",
        "NLP helps computers understand and process human language.",
        "Computer vision allows machines to see and understand images."
    ]
    
    ground_truths = [
        "Deep learning is a machine learning technique using multi-layered neural networks.",
        "NLP is AI technology for processing and understanding human language.",
        "Computer vision enables computers to interpret visual data."
    ]
    
    # Evaluate batch
    results = evaluator.evaluate_batch(
        queries=queries,
        contexts=contexts,
        answers=answers,
        ground_truths=ground_truths
    )
    
    # Display results
    print(f"\nEvaluated {len(results)} examples")
    print("\n" + "-" * 60)
    
    for i, result in enumerate(results):
        print(f"\nExample {i+1}: {queries[i]}")
        for metric_name, metric_result in result.items():
            score = metric_result.get('score', 'N/A')
            if score is not None:
                print(f"  {metric_name}: {score:.3f}")
            else:
                print(f"  {metric_name}: N/A")
    
    # Get average scores
    avg_scores = evaluator.get_average_scores(results)
    print("\n" + "=" * 60)
    print("AVERAGE SCORES:")
    print("=" * 60)
    for metric_name, score in avg_scores.items():
        if score is not None:
            print(f"  {metric_name}: {score:.3f}")
        else:
            print(f"  {metric_name}: N/A")
    
    print("\n")


def example_custom_metrics():
    """Example of using specific metrics only."""
    print("=" * 60)
    print("Example 3: Custom Metrics Selection")
    print("=" * 60)
    
    # Initialize evaluator with only faithfulness and relevance
    evaluator = RAGEvaluator(metrics=['faithfulness', 'relevance'])
    
    query = "What is reinforcement learning?"
    context = "Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize rewards."
    answer = "Reinforcement learning is when AI learns by trial and error to maximize rewards."
    
    results = evaluator.evaluate(
        query=query,
        context=context,
        answer=answer
    )
    
    print(f"\nQuery: {query}")
    print(f"\nAnswer: {answer}")
    print("\n" + "-" * 60)
    print(f"Using metrics: {list(evaluator.metrics.keys())}")
    print("-" * 60)
    
    for metric_name, result in results.items():
        print(f"\n{metric_name.upper()}: {result['score']:.3f}")
    
    print("\n")


def example_data_loading():
    """Example of loading data from files."""
    print("=" * 60)
    print("Example 4: Data Loading (CSV format)")
    print("=" * 60)
    
    # This is a demo - in real usage, you would load from actual files
    print("\nDataTableLoader supports:")
    print("  - CSV files")
    print("  - JSON files")
    print("  - Excel files (.xlsx, .xls)")
    
    print("\nJabrefLoader supports:")
    print("  - BibTeX files")
    print("  - JSON files")
    
    print("\nExample usage:")
    print("  loader = DataTableLoader()")
    print("  data = loader.load_for_evaluation('data.csv')")
    print("  evaluator = RAGEvaluator()")
    print("  results = evaluator.evaluate_batch(**data)")
    
    print("\n")


if __name__ == "__main__":
    print("\n")
    print("*" * 60)
    print("RAG EVALUATION FRAMEWORK - EXAMPLES")
    print("*" * 60)
    print("\n")
    
    # Run examples
    example_single_evaluation()
    example_batch_evaluation()
    example_custom_metrics()
    example_data_loading()
    
    print("*" * 60)
    print("All examples completed!")
    print("*" * 60)
    print("\n")
