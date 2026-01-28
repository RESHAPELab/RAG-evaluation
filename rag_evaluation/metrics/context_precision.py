"""
Context Precision Metric

Measures how much of the information in the answer actually comes from
the correct ground truth context, as opposed to irrelevant or incorrect
context.
"""

import re
from typing import Dict, Any, Set
from . import BaseMetric
from .utils import STOP_WORDS


def extract_key_terms(text: str) -> Set[str]:
    """
    Extract key terms from text by removing stop words and punctuation.
    
    Args:
        text: Input text
        
    Returns:
        Set of key terms
    """
    # Convert to lowercase and remove punctuation
    text_lower = text.lower()
    text_clean = re.sub(r'[^\w\s]', ' ', text_lower)
    
    # Split into words
    words = text_clean.split()
    
    # Filter stop words and short words
    key_terms = {word for word in words if word not in STOP_WORDS and len(word) > 2}
    
    return key_terms


class ContextPrecisionMetric(BaseMetric):
    """
    Metric to evaluate context precision.
    
    Context precision measures how much of the information in the answer
    comes from the ground truth context. It evaluates whether the model
    is using the right parts of the context to generate answers.
    
    Score ranges from 0.0 to 1.0, where:
    - 1.0 indicates answer is fully based on ground truth context
    - 0.0 indicates answer has no overlap with ground truth context
    """
    
    def compute(self, answer: str, context: str, ground_truth: str) -> Dict[str, Any]:
        """
        Compute context precision score.
        
        Args:
            answer: The generated answer
            context: The retrieved context used to generate the answer
            ground_truth: The ground truth context or answer
            
        Returns:
            Dictionary containing:
                - score: Context precision score (0.0 to 1.0)
                - details: Additional information about the evaluation
        """
        # Extract key terms from each text
        answer_terms = extract_key_terms(answer)
        ground_truth_terms = extract_key_terms(ground_truth)
        context_terms = extract_key_terms(context)
        
        if not answer_terms:
            return {
                'score': 1.0,
                'details': {
                    'reasoning': 'Empty answer, no terms to evaluate',
                    'answer_terms_count': 0,
                    'ground_truth_overlap': 0,
                    'context_overlap': 0
                }
            }
        
        # Calculate overlap between answer and ground truth
        answer_gt_overlap = answer_terms.intersection(ground_truth_terms)
        
        # Calculate overlap between answer and context
        answer_context_overlap = answer_terms.intersection(context_terms)
        
        # Precision: how many answer terms come from ground truth
        if answer_terms:
            precision_score = len(answer_gt_overlap) / len(answer_terms)
        else:
            precision_score = 0.0
        
        return {
            'score': precision_score,
            'details': {
                'answer_terms_count': len(answer_terms),
                'ground_truth_overlap': len(answer_gt_overlap),
                'context_overlap': len(answer_context_overlap),
                'precision_percentage': f'{precision_score * 100:.1f}%',
                'reasoning': f'{len(answer_gt_overlap)} out of {len(answer_terms)} answer terms found in ground truth'
            }
        }
