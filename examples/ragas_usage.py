"""
Example usage of the RAG Evaluation Framework with Ragas Integration

This script demonstrates how to use the ragas library for advanced
RAG evaluation with LLM-based metrics.
"""

import sys
from pathlib import Path
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_ragas_available():
    """Check if ragas is available and properly configured."""
    try:
        from rag_evaluation import RagasEvaluator
        return True
    except ImportError:
        print("ERROR: ragas library is not installed.")
        print("Install it with: pip install ragas")
        return False


def example_ragas_single_evaluation():
    """Example of evaluating a single RAG output using ragas."""
    print("=" * 60)
    print("Example: Single Evaluation with Ragas")
    print("=" * 60)
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\nWARNING: OPENAI_API_KEY environment variable not set.")
        print("Ragas requires an OpenAI API key for evaluation.")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        print("Skipping ragas example...\n")
        return
    
    try:
        from rag_evaluation import RagasEvaluator
        
        # Initialize ragas evaluator
        evaluator = RagasEvaluator()
        
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
        
        # Evaluate using ragas
        print(f"\nQuery: {query}")
        print(f"\nAnswer: {answer}")
        print("\n" + "-" * 60)
        print("Evaluating with ragas library...")
        print("-" * 60)
        
        results = evaluator.evaluate(
            query=query,
            context=context,
            answer=answer,
            ground_truth=ground_truth
        )
        
        # Display results
        print("\nEVALUATION RESULTS (using ragas):")
        print("-" * 60)
        
        for metric_name, result in results.items():
            print(f"\n{metric_name.upper()}:")
            score = result.get('score')
            if score is not None:
                print(f"  Score: {score:.3f}")
                if 'details' in result:
                    print(f"  Library: {result['details'].get('library', 'N/A')}")
            else:
                error = result.get('error', 'Unknown error')
                print(f"  Error: {error}")
        
        print("\n")
        
    except Exception as e:
        print(f"\nError running ragas evaluation: {e}")
        print("Make sure OPENAI_API_KEY is set and valid.\n")


def example_ragas_batch_evaluation():
    """Example of evaluating multiple RAG outputs using ragas."""
    print("=" * 60)
    print("Example: Batch Evaluation with Ragas")
    print("=" * 60)
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\nWARNING: OPENAI_API_KEY environment variable not set.")
        print("Skipping ragas batch example...\n")
        return
    
    try:
        from rag_evaluation import RagasEvaluator
        
        # Initialize ragas evaluator
        evaluator = RagasEvaluator()
        
        # Example batch data
        queries = [
            "What is deep learning?",
            "What is natural language processing?",
        ]
        
        contexts = [
            "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
            "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language.",
        ]
        
        answers = [
            "Deep learning uses multi-layer neural networks to learn from data.",
            "NLP helps computers understand and process human language.",
        ]
        
        ground_truths = [
            "Deep learning is a machine learning technique using multi-layered neural networks.",
            "NLP is AI technology for processing and understanding human language.",
        ]
        
        # Evaluate batch using ragas
        print(f"\nEvaluating {len(queries)} examples with ragas...")
        print("-" * 60)
        
        results = evaluator.evaluate_batch(
            queries=queries,
            contexts=contexts,
            answers=answers,
            ground_truths=ground_truths
        )
        
        # Display results
        print("\nINDIVIDUAL RESULTS:")
        print("-" * 60)
        
        for i, result in enumerate(results):
            print(f"\nExample {i+1}: {queries[i]}")
            for metric_name, metric_result in result.items():
                score = metric_result.get('score', 'N/A')
                if score is not None and score != 'N/A':
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
        
    except Exception as e:
        print(f"\nError running ragas batch evaluation: {e}\n")


def example_comparison():
    """Compare basic evaluator with ragas evaluator."""
    print("=" * 60)
    print("Example: Comparing Basic vs Ragas Evaluators")
    print("=" * 60)
    
    from rag_evaluation import RAGEvaluator
    
    # Example data
    query = "What is reinforcement learning?"
    context = (
        "Reinforcement learning is a type of machine learning where an agent "
        "learns to make decisions by taking actions in an environment to maximize rewards."
    )
    answer = "Reinforcement learning is when AI learns by trial and error to maximize rewards."
    
    # Basic evaluator
    print("\n1. Using Basic Evaluator (rule-based metrics):")
    print("-" * 60)
    basic_evaluator = RAGEvaluator()
    basic_results = basic_evaluator.evaluate(query, context, answer)
    
    for metric_name, result in basic_results.items():
        score = result.get('score')
        if score is not None:
            print(f"  {metric_name}: {score:.3f}")
    
    # Ragas evaluator
    if os.environ.get("OPENAI_API_KEY"):
        print("\n2. Using Ragas Evaluator (LLM-based metrics):")
        print("-" * 60)
        try:
            from rag_evaluation import RagasEvaluator
            ragas_evaluator = RagasEvaluator()
            ragas_results = ragas_evaluator.evaluate(query, context, answer)
            
            for metric_name, result in ragas_results.items():
                score = result.get('score')
                if score is not None:
                    print(f"  {metric_name}: {score:.3f}")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print("\n2. Ragas Evaluator: Skipped (OPENAI_API_KEY not set)")
    
    print("\n")


if __name__ == "__main__":
    print("\n")
    print("*" * 60)
    print("RAG EVALUATION FRAMEWORK - RAGAS INTEGRATION EXAMPLES")
    print("*" * 60)
    print("\n")
    
    # Check if ragas is available
    if not check_ragas_available():
        sys.exit(1)
    
    print("Ragas library is available!\n")
    
    # Run examples
    example_ragas_single_evaluation()
    example_ragas_batch_evaluation()
    example_comparison()
    
    print("*" * 60)
    print("All examples completed!")
    print("*" * 60)
    print("\n")
