#!/usr/bin/env python3
"""
RAG Model Evaluation Runner

This script provides a command-line interface for evaluating RAG models
using the evaluation framework. It can load data from various formats
and output detailed evaluation results.
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path to import rag_evaluation
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_evaluation import RAGEvaluator
from rag_evaluation.data_ingestion import DataTableLoader, JabrefLoader


def load_data(file_path, data_type='auto'):
    """
    Load data from file using appropriate loader.
    
    Args:
        file_path: Path to data file
        data_type: Type of data ('csv', 'json', 'bibtex', or 'auto')
        
    Returns:
        Dictionary with queries, contexts, answers, and ground_truths
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Auto-detect type if needed
    if data_type == 'auto':
        ext = path.suffix.lower()
        if ext == '.csv':
            data_type = 'csv'
        elif ext == '.json':
            data_type = 'json'
        elif ext in ['.bib', '.bibtex']:
            data_type = 'bibtex'
        else:
            raise ValueError(f"Cannot auto-detect type for extension: {ext}")
    
    # Load data
    if data_type == 'bibtex':
        loader = JabrefLoader()
    else:
        loader = DataTableLoader()
    
    return loader.load_for_evaluation(file_path)


def print_results(results, avg_scores, verbose=False):
    """
    Print evaluation results in a formatted way.
    
    Args:
        results: List of evaluation results
        avg_scores: Average scores across all examples
        verbose: Whether to print detailed results for each example
    """
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    if verbose:
        print(f"\nEvaluated {len(results)} examples:\n")
        for i, result in enumerate(results, 1):
            print(f"Example {i}:")
            for metric_name, metric_result in result.items():
                score = metric_result.get('score', 'N/A')
                if score is not None:
                    print(f"  {metric_name}: {score:.3f}")
                    if 'details' in metric_result:
                        reasoning = metric_result['details'].get('reasoning', '')
                        if reasoning:
                            print(f"    └─ {reasoning}")
                else:
                    print(f"  {metric_name}: N/A")
            print()
    
    # Print average scores
    print("\n" + "-" * 70)
    print(f"AVERAGE SCORES (across {len(results)} examples)")
    print("-" * 70)
    
    for metric_name, score in avg_scores.items():
        if score is not None:
            print(f"  {metric_name.upper()}: {score:.3f}")
            
            # Add interpretation
            if score >= 0.8:
                level = "Excellent"
            elif score >= 0.6:
                level = "Good"
            elif score >= 0.4:
                level = "Fair"
            else:
                level = "Needs Improvement"
            
            print(f"    └─ {level}")
        else:
            print(f"  {metric_name.upper()}: N/A")
    
    print("\n" + "=" * 70)


def save_results(results, avg_scores, output_file):
    """
    Save evaluation results to a JSON file.
    
    Args:
        results: List of evaluation results
        avg_scores: Average scores
        output_file: Path to output file
    """
    output_data = {
        'individual_results': results,
        'average_scores': avg_scores,
        'num_examples': len(results)
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Evaluate RAG models using faithfulness, context precision, and relevance metrics.'
    )
    
    parser.add_argument(
        'data_file',
        help='Path to the data file (CSV, JSON, or BibTeX format)'
    )
    
    parser.add_argument(
        '--type',
        choices=['csv', 'json', 'bibtex', 'auto'],
        default='auto',
        help='Type of data file (default: auto-detect from extension)'
    )
    
    parser.add_argument(
        '--metrics',
        nargs='+',
        choices=['faithfulness', 'context_precision', 'relevance'],
        help='Specific metrics to evaluate (default: all metrics)'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        help='Save results to JSON file'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Show detailed results for each example'
    )
    
    args = parser.parse_args()
    
    try:
        # Load data
        print(f"Loading data from: {args.data_file}")
        data = load_data(args.data_file, args.type)
        print(f"Loaded {len(data['queries'])} examples")
        
        # Initialize evaluator
        evaluator = RAGEvaluator(metrics=args.metrics)
        print(f"Using metrics: {list(evaluator.metrics.keys())}")
        
        # Run evaluation
        print("\nRunning evaluation...")
        results = evaluator.evaluate_batch(**data)
        avg_scores = evaluator.get_average_scores(results)
        
        # Display results
        print_results(results, avg_scores, verbose=args.verbose)
        
        # Save results if requested
        if args.output:
            save_results(results, avg_scores, args.output)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
