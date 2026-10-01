"""Base scorer interface."""

from abc import ABC, abstractmethod
from typing import Any


class BaseScorer(ABC):
    """Abstract base class for all scorers."""
    
    @abstractmethod
    def score(self, prompt: str, response: str, reference: dict[str, Any]) -> float:
        """Score a single response.
        
        Args:
            prompt: The input prompt
            response: The model's response
            reference: Reference data from the dataset
            
        Returns:
            Score between 0.0 and 1.0
        """
        pass
    
    @abstractmethod
    def is_correct(self, prompt: str, response: str, reference: dict[str, Any]) -> bool:
        """Check if response is correct (score == 1.0).
        
        This is used for flip detection - we need binary correct/incorrect.
        """
        pass
