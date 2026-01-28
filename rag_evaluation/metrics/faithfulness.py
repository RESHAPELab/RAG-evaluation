"""
Faithfulness Metric

Evaluates whether the generated answer is grounded in the provided context
and not hallucinated. This metric checks if claims in the answer can be
verified from the context.
"""

import re
from typing import Dict, List, Any
from . import BaseMetric
from .utils import STOP_WORDS, FAITHFULNESS_SUPPORT_THRESHOLD


class FaithfulnessMetric(BaseMetric):
    """
    Metric to evaluate faithfulness of answers to the provided context.
    
    Faithfulness measures whether the answer is grounded in the provided
    context rather than containing hallucinated information. It checks if
    the statements in the answer can be supported by the context.
    
    Score ranges from 0.0 to 1.0, where:
    - 1.0 indicates all statements are supported by context
    - 0.0 indicates no statements are supported by context
    """
    
    def compute(self, answer: str, context: str) -> Dict[str, Any]:
        """
        Compute faithfulness score for the answer given the context.
        
        Args:
            answer: The generated answer to evaluate
            context: The retrieved context used to generate the answer
            
        Returns:
            Dictionary containing:
                - score: Faithfulness score (0.0 to 1.0)
                - details: Additional information about the evaluation
        """
        # Split answer into sentences for analysis
        answer_sentences = self._split_into_sentences(answer)
        context_lower = context.lower()
        
        if not answer_sentences:
            return {
                'score': 1.0,
                'details': {
                    'total_sentences': 0,
                    'supported_sentences': 0,
                    'reasoning': 'Empty answer, no claims to verify'
                }
            }
        
        # Check how many sentences have support in context
        supported_count = 0
        unsupported_sentences = []
        
        for sentence in answer_sentences:
            if self._is_supported_by_context(sentence, context_lower):
                supported_count += 1
            else:
                unsupported_sentences.append(sentence)
        
        # Calculate faithfulness score
        score = supported_count / len(answer_sentences)
        
        return {
            'score': score,
            'details': {
                'total_sentences': len(answer_sentences),
                'supported_sentences': supported_count,
                'unsupported_sentences': unsupported_sentences,
                'reasoning': f'{supported_count} out of {len(answer_sentences)} sentences are grounded in context'
            }
        }
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be enhanced with NLP libraries)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _is_supported_by_context(self, sentence: str, context_lower: str) -> bool:
        """
        Check if a sentence is supported by the context.
        
        This is a basic implementation that checks for keyword overlap.
        In production, this could use more sophisticated NLP techniques.
        
        Args:
            sentence: Sentence to check
            context_lower: Context in lowercase
            
        Returns:
            True if sentence appears supported by context
        """
        # Extract key terms from sentence (remove common words)
        sentence_lower = sentence.lower()
        words = sentence_lower.split()
        key_words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
        
        # Check if majority of key words appear in context
        if not key_words:
            return True  # No meaningful words to verify
        
        found_count = sum(1 for word in key_words if word in context_lower)
        
        # Consider supported if threshold % of key words are in context
        return found_count / len(key_words) > FAITHFULNESS_SUPPORT_THRESHOLD
