"""
Ragas-based RAG Evaluator

This module provides integration with the ragas library for advanced RAG evaluation
using LLM-based metrics. Ragas provides more sophisticated evaluation metrics
compared to the basic rule-based metrics.
"""

from typing import Dict, List, Any, Optional
import os

try:
    from ragas import SingleTurnSample, EvaluationDataset, evaluate
    from ragas.metrics.collections.faithfulness import Faithfulness
    from ragas.metrics.collections.answer_relevancy import AnswerRelevancy
    from ragas.metrics.collections.context_precision import ContextPrecision
    from ragas.metrics.collections.context_recall import ContextRecall
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


class RagasEvaluator:
    """
    RAG evaluator using the ragas library for LLM-based evaluation metrics.
    
    This evaluator uses ragas library which provides advanced metrics including:
    - Faithfulness: Measures factual consistency of the answer with the context
    - Answer Relevancy: Measures how relevant the answer is to the query
    - Context Precision: Measures how relevant the retrieved context is
    - Context Recall: Measures if all relevant information is retrieved
    
    Note: This evaluator requires an LLM (OpenAI by default) to compute metrics.
    Set OPENAI_API_KEY environment variable to use this evaluator.
    
    Example:
        >>> evaluator = RagasEvaluator()
        >>> results = evaluator.evaluate(
        ...     query="What is machine learning?",
        ...     context="Machine learning is a subset of AI...",
        ...     answer="Machine learning allows computers to learn...",
        ...     ground_truth="Machine learning is a subset of AI..."
        ... )
    """
    
    def __init__(self, metrics: Optional[List[str]] = None, llm=None):
        """
        Initialize the Ragas evaluator with specified metrics.
        
        Args:
            metrics: List of metric names to use. If None, uses all metrics.
                    Options: ['faithfulness', 'answer_relevancy', 
                             'context_precision', 'context_recall']
            llm: Optional LLM instance to use for evaluation. If None, uses default.
        
        Raises:
            ImportError: If ragas library is not installed
            ValueError: If OPENAI_API_KEY is not set
        """
        if not RAGAS_AVAILABLE:
            raise ImportError(
                "ragas library is not installed. "
                "Install it with: pip install ragas"
            )
        
        # Check for API key
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY environment variable must be set to use RagasEvaluator. "
                "Get your API key from https://platform.openai.com/api-keys"
            )
        
        # Store metric names for validation
        self.available_metric_names = [
            'faithfulness',
            'answer_relevancy',
            'context_precision',
            'context_recall'
        ]
        
        if metrics is None:
            self.metric_names = self.available_metric_names
        else:
            # Validate metric names
            invalid_metrics = [m for m in metrics if m not in self.available_metric_names]
            if invalid_metrics:
                raise ValueError(
                    f"Invalid metric names: {invalid_metrics}. "
                    f"Valid options are: {self.available_metric_names}"
                )
            self.metric_names = metrics
        
        # Store LLM for later use
        self.llm = llm
        
    def _get_metrics(self):
        """Get metric instances. Created on demand to avoid issues with LLM initialization."""
        # Map metric names to ragas metric classes
        # Note: metrics are created without llm parameter - ragas will use default
        available_metrics = {
            'faithfulness': Faithfulness,
            'answer_relevancy': AnswerRelevancy,
            'context_precision': ContextPrecision,
            'context_recall': ContextRecall
        }
        
        # Get the actual metric objects (instances, not classes)
        metrics = [
            available_metrics[name]() if self.llm is None else available_metrics[name](llm=self.llm)
            for name in self.metric_names
        ]
        
        return metrics
    
    def evaluate(
        self,
        query: str,
        context: str,
        answer: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single RAG model output using ragas metrics.
        
        Args:
            query: The user's question or query
            context: The retrieved context used to generate the answer
            answer: The generated answer from the RAG model
            ground_truth: Optional ground truth answer for comparison
            
        Returns:
            Dictionary containing scores for each metric
        """
        # Create a ragas sample
        # Note: ragas expects retrieved_contexts as a list
        contexts_list = [context] if isinstance(context, str) else context
        
        sample = SingleTurnSample(
            user_input=query,
            retrieved_contexts=contexts_list,
            response=answer,
            reference=ground_truth
        )
        
        # Create dataset with single sample
        dataset = EvaluationDataset(samples=[sample])
        
        # Get metrics
        metrics = self._get_metrics()
        
        # Evaluate using ragas
        # Set show_progress=False for single evaluations
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            show_progress=False,
            raise_exceptions=True
        )
        
        # Convert result to dictionary format
        results = {}
        for metric_name in self.metric_names:
            if metric_name in result:
                score_value = result[metric_name]
                # Handle cases where score might be a list
                if isinstance(score_value, list) and len(score_value) > 0:
                    score_value = score_value[0]
                
                results[metric_name] = {
                    'score': float(score_value) if score_value is not None else None,
                    'details': {
                        'reasoning': f'Evaluated using ragas {metric_name} metric',
                        'library': 'ragas'
                    }
                }
            else:
                results[metric_name] = {
                    'score': None,
                    'error': f'Metric {metric_name} not computed'
                }
        
        return results
    
    def evaluate_batch(
        self,
        queries: List[str],
        contexts: List[str],
        answers: List[str],
        ground_truths: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple RAG model outputs in batch using ragas.
        
        Args:
            queries: List of queries
            contexts: List of contexts
            answers: List of generated answers
            ground_truths: Optional list of ground truth answers
            
        Returns:
            List of evaluation results for each example
        
        Raises:
            ValueError: If input lists have different lengths
        """
        # Validate input lengths
        if not (len(queries) == len(contexts) == len(answers)):
            raise ValueError(
                f"Input lists must have the same length. "
                f"Got queries={len(queries)}, contexts={len(contexts)}, answers={len(answers)}"
            )
        
        if ground_truths is None:
            ground_truths = [None] * len(queries)
        elif len(ground_truths) != len(queries):
            raise ValueError(
                f"ground_truths length ({len(ground_truths)}) must match queries length ({len(queries)})"
            )
        
        # Create ragas samples
        samples = []
        for query, context, answer, gt in zip(queries, contexts, answers, ground_truths):
            # ragas expects retrieved_contexts as a list
            contexts_list = [context] if isinstance(context, str) else context
            
            sample = SingleTurnSample(
                user_input=query,
                retrieved_contexts=contexts_list,
                response=answer,
                reference=gt
            )
            samples.append(sample)
        
        # Create dataset
        dataset = EvaluationDataset(samples=samples)
        
        # Get metrics
        metrics = self._get_metrics()
        
        # Evaluate using ragas
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            show_progress=True,
            raise_exceptions=True
        )
        
        # Convert results to list of dictionaries
        batch_results = []
        for i in range(len(samples)):
            sample_result = {}
            for metric_name in self.metric_names:
                if metric_name in result:
                    scores = result[metric_name]
                    score_value = scores[i] if isinstance(scores, list) else scores
                    
                    sample_result[metric_name] = {
                        'score': float(score_value) if score_value is not None else None,
                        'details': {
                            'reasoning': f'Evaluated using ragas {metric_name} metric',
                            'library': 'ragas'
                        }
                    }
                else:
                    sample_result[metric_name] = {
                        'score': None,
                        'error': f'Metric {metric_name} not computed'
                    }
            batch_results.append(sample_result)
        
        return batch_results
    
    def get_average_scores(self, batch_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute average scores across batch results.
        
        Args:
            batch_results: List of evaluation results from evaluate_batch
            
        Returns:
            Dictionary with average scores for each metric
        """
        average_scores = {}
        
        for metric_name in self.metric_names:
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
