"""
Main RAG Evaluator class that orchestrates the evaluation process.
"""

from typing import Dict, List, Any, Optional
from .metrics.faithfulness import FaithfulnessMetric
from .metrics.context_precision import ContextPrecisionMetric
from .metrics.relevance import RelevanceMetric


class RAGEvaluator:
    """
    Main evaluator for RAG models that computes multiple evaluation metrics.
    
    This class provides a unified interface to evaluate RAG model outputs
    across three key dimensions:
    - Faithfulness: Whether the answer is grounded in context
    - Context Precision: How much information comes from ground truth
    - Relevance: Whether the response is relevant to query and context
    
    Example:
        >>> evaluator = RAGEvaluator()
        >>> results = evaluator.evaluate(
        ...     query="What is machine learning?",
        ...     context="Machine learning is a subset of AI...",
        ...     answer="Machine learning is about teaching computers...",
        ...     ground_truth="Machine learning is a subset of AI..."
        ... )
    """
    
    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Initialize the RAG evaluator with specified metrics.
        
        Args:
            metrics: List of metric names to use. If None, uses all metrics.
                    Options: ['faithfulness', 'context_precision', 'relevance']
        """
        self.available_metrics = {
            'faithfulness': FaithfulnessMetric(),
            'context_precision': ContextPrecisionMetric(),
            'relevance': RelevanceMetric()
        }
        
        if metrics is None:
            self.metrics = self.available_metrics
        else:
            self.metrics = {
                name: metric 
                for name, metric in self.available_metrics.items() 
                if name in metrics
            }
    
    def evaluate(
        self,
        query: str,
        context: str,
        answer: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single RAG model output.
        
        Args:
            query: The user's question or query
            context: The retrieved context used to generate the answer
            answer: The generated answer from the RAG model
            ground_truth: Optional ground truth answer for comparison
            
        Returns:
            Dictionary containing scores for each metric
        """
        results = {}
        
        for metric_name, metric in self.metrics.items():
            if metric_name == 'faithfulness':
                results[metric_name] = metric.compute(answer, context)
            elif metric_name == 'context_precision':
                if ground_truth:
                    results[metric_name] = metric.compute(answer, context, ground_truth)
                else:
                    results[metric_name] = {
                        'score': None,
                        'error': 'Ground truth required for context precision'
                    }
            elif metric_name == 'relevance':
                results[metric_name] = metric.compute(query, answer, context)
        
        return results
    
    def evaluate_batch(
        self,
        queries: List[str],
        contexts: List[str],
        answers: List[str],
        ground_truths: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple RAG model outputs in batch.
        
        Args:
            queries: List of queries
            contexts: List of contexts
            answers: List of generated answers
            ground_truths: Optional list of ground truth answers
            
        Returns:
            List of evaluation results for each example
        """
        if ground_truths is None:
            ground_truths = [None] * len(queries)
        
        results = []
        for query, context, answer, gt in zip(queries, contexts, answers, ground_truths):
            result = self.evaluate(query, context, answer, gt)
            results.append(result)
        
        return results
    
    def get_average_scores(self, batch_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute average scores across batch results.
        
        Args:
            batch_results: List of evaluation results from evaluate_batch
            
        Returns:
            Dictionary with average scores for each metric
        """
        average_scores = {}
        
        for metric_name in self.metrics.keys():
            scores = []
            for result in batch_results:
                if metric_name in result:
                    metric_result = result[metric_name]
                    if isinstance(metric_result, dict) and 'score' in metric_result:
                        if metric_result['score'] is not None:
                            scores.append(metric_result['score'])
            
            if scores:
                average_scores[metric_name] = sum(scores) / len(scores)
            else:
                average_scores[metric_name] = None
        
        return average_scores
