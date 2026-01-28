"""
Relevance Metric

Checks if the generated response is actually relevant to both the query
and the provided context.
"""

from typing import Dict, Any, Set
from . import BaseMetric


class RelevanceMetric(BaseMetric):
    """
    Metric to evaluate relevance of the answer to query and context.
    
    Relevance measures whether the generated response appropriately addresses
    the user's query and makes use of the provided context. It checks if the
    answer is on-topic and responsive.
    
    Score ranges from 0.0 to 1.0, where:
    - 1.0 indicates highly relevant answer
    - 0.0 indicates completely irrelevant answer
    """
    
    def __init__(self):
        """Initialize the relevance metric."""
        pass
    
    def compute(self, query: str, answer: str, context: str) -> Dict[str, Any]:
        """
        Compute relevance score for the answer given query and context.
        
        Args:
            query: The user's question or query
            answer: The generated answer
            context: The retrieved context
            
        Returns:
            Dictionary containing:
                - score: Relevance score (0.0 to 1.0)
                - details: Additional information about the evaluation
        """
        # Extract key terms from each component
        query_terms = self._extract_key_terms(query)
        answer_terms = self._extract_key_terms(answer)
        context_terms = self._extract_key_terms(context)
        
        if not query_terms or not answer_terms:
            return {
                'score': 0.0,
                'details': {
                    'reasoning': 'Empty query or answer',
                    'query_terms_count': len(query_terms),
                    'answer_terms_count': len(answer_terms)
                }
            }
        
        # Calculate query-answer relevance (how well answer addresses query)
        query_answer_overlap = query_terms.intersection(answer_terms)
        query_relevance = len(query_answer_overlap) / len(query_terms)
        
        # Calculate context-answer relevance (how well answer uses context)
        context_answer_overlap = context_terms.intersection(answer_terms)
        if context_terms:
            context_relevance = len(context_answer_overlap) / len(context_terms)
        else:
            context_relevance = 0.0
        
        # Combined relevance score (weighted average)
        # Give more weight to query relevance (70%) vs context relevance (30%)
        relevance_score = (0.7 * query_relevance) + (0.3 * context_relevance)
        
        return {
            'score': relevance_score,
            'details': {
                'query_relevance': query_relevance,
                'context_relevance': context_relevance,
                'query_terms_in_answer': len(query_answer_overlap),
                'total_query_terms': len(query_terms),
                'context_terms_in_answer': len(context_answer_overlap),
                'total_context_terms': len(context_terms),
                'reasoning': f'Answer addresses {len(query_answer_overlap)}/{len(query_terms)} query terms and uses {len(context_answer_overlap)}/{len(context_terms)} context terms'
            }
        }
    
    def _extract_key_terms(self, text: str) -> Set[str]:
        """
        Extract key terms from text by removing stop words and punctuation.
        
        Args:
            text: Input text
            
        Returns:
            Set of key terms
        """
        import re
        
        # Convert to lowercase and remove punctuation
        text_lower = text.lower()
        text_clean = re.sub(r'[^\w\s]', ' ', text_lower)
        
        # Split into words
        words = text_clean.split()
        
        # Remove stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
            'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 
            'was', 'were', 'be', 'been', 'being', 'this', 'that',
            'these', 'those', 'it', 'its', 'they', 'their', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'which', 'who', 'what',
            'where', 'when', 'why', 'how'
        }
        
        # Filter stop words and short words
        key_terms = {word for word in words if word not in stop_words and len(word) > 2}
        
        return key_terms
