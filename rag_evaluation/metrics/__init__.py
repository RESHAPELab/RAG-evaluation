"""
Base class for all evaluation metrics.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseMetric(ABC):
    """
    Abstract base class for evaluation metrics.
    """
    
    @abstractmethod
    def compute(self, *args, **kwargs) -> Any:
        """
        Compute the metric score.
        
        Returns:
            Metric result (typically a dictionary with score and details)
        """
        pass
