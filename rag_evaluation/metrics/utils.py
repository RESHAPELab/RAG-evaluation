"""
Utility functions shared across metrics.
"""

# Common stop words used for text analysis across all metrics
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
    'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'this', 'that',
    'these', 'those', 'it', 'its', 'they', 'their', 'have',
    'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'which', 'who', 'what',
    'where', 'when', 'why', 'how'
}

# Threshold for determining if a sentence is supported by context
# A sentence is considered supported if >50% of its key terms appear in the context
FAITHFULNESS_SUPPORT_THRESHOLD = 0.5

# Weights for computing relevance score
# Higher weight on query relevance (does answer address the question?)
# Lower weight on context relevance (does answer use the context?)
RELEVANCE_QUERY_WEIGHT = 0.7
RELEVANCE_CONTEXT_WEIGHT = 0.3
