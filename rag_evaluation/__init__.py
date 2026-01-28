"""
RAG Evaluation Framework

A framework for evaluating Retrieval-Augmented Generation (RAG) models
using multiple metrics including faithfulness, context precision, and relevance.
"""

from .evaluator import RAGEvaluator
from .metrics.faithfulness import FaithfulnessMetric
from .metrics.context_precision import ContextPrecisionMetric
from .metrics.relevance import RelevanceMetric

# Try to import RagasEvaluator (optional dependency)
try:
    from .ragas_evaluator import RagasEvaluator
    _RAGAS_AVAILABLE = True
    __all__ = [
        "RAGEvaluator",
        "FaithfulnessMetric",
        "ContextPrecisionMetric",
        "RelevanceMetric",
        "RagasEvaluator",
    ]
except ImportError:
    _RAGAS_AVAILABLE = False
    # Don't export RagasEvaluator if not available
    __all__ = [
        "RAGEvaluator",
        "FaithfulnessMetric",
        "ContextPrecisionMetric",
        "RelevanceMetric",
    ]

__version__ = "0.1.0"
