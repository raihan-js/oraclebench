"""FedProc registry hallucination scorer: checks clause numbers against real FAR/DFARS registry."""

import re
from typing import Any

from .base import BaseScorer


class FedProcRegistryScorer(BaseScorer):
    """Score by checking if cited clause numbers exist in the real FAR/DFARS registry.
    
    This scorer checks for hallucinated clause numbers. A response is correct if:
    1. It cites clause numbers that exist in the registry
    2. It does not cite clause numbers that don't exist
    
    Uses only the real-FAR slice (155 records) from FedProc-Bench.
    """
    
    def __init__(self, registry: set[str] | None = None):
        """Initialize with a set of valid clause numbers.
        
        Args:
            registry: Set of valid clause numbers (e.g., {"52.203-1", "52.212-4"})
                     If None, must be loaded from data.
        """
        self.registry = registry or set()
    
    @classmethod
    def from_file(cls, path: str) -> "FedProcRegistryScorer":
        """Load registry from a file (one clause per line)."""
        with open(path) as f:
            registry = {line.strip() for line in f if line.strip()}
        return cls(registry=registry)
    
    def extract_clause_numbers(self, response: str) -> list[str]:
        """Extract FAR/DFARS clause numbers from response.
        
        Patterns: FAR 52.203-1, DFARS 252.203-1, etc.
        """
        # Match patterns like "FAR 52.203-1" or "DFARS 252.212-4"
        pattern = r"(?:FAR|DFARS)\s+(\d{2,3}\.\d{3}-\d+)"
        matches = re.findall(pattern, response, re.IGNORECASE)
        return matches
    
    def score(self, prompt: str, response: str, reference: dict[str, Any]) -> float:
        """Score based on whether cited clauses exist in registry.
        
        Returns 1.0 if all cited clauses are valid, 0.0 if any are hallucinated.
        If no clauses are cited, returns 1.0 (no hallucination detected).
        """
        cited_clauses = self.extract_clause_numbers(response)
        
        if not cited_clauses:
            # No clauses cited - no hallucination detected
            return 1.0
        
        # Check if all cited clauses exist in registry
        invalid_clauses = [c for c in cited_clauses if c not in self.registry]
        
        if invalid_clauses:
            return 0.0
        
        return 1.0
    
    def is_correct(self, prompt: str, response: str, reference: dict[str, Any]) -> bool:
        """Check if response has no hallucinated clause numbers."""
        return self.score(prompt, response, reference) == 1.0
    
    def get_hallucinated_clauses(self, response: str) -> list[str]:
        """Return list of hallucinated clause numbers (for reporting)."""
        cited_clauses = self.extract_clause_numbers(response)
        return [c for c in cited_clauses if c not in self.registry]
