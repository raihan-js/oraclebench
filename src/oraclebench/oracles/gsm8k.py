"""GSM8K scorer: exact numeric match on final answer."""

import re
from typing import Any

from .base import BaseScorer


class GSM8KScorer(BaseScorer):
    """Score GSM8K responses by extracting the final numeric answer."""
    
    # Pattern to find the final answer (usually after "####" or "The answer is")
    ANSWER_PATTERNS = [
        r"####\s*(\d+\.?\d*)",  # #### 42
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\s]+\$?(\d+\.?\d*)",  # The answer is 42
        r"=\s*\$?(\d+\.?\d*)\s*$",  # = 42 (at end)
    ]
    
    def extract_answer(self, response: str) -> float | None:
        """Extract the final numeric answer from response."""
        for pattern in self.ANSWER_PATTERNS:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        # Fallback: find last number in response
        numbers = re.findall(r"\$?(\d+\.?\d*)", response)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        
        return None
    
    def score(self, prompt: str, response: str, reference: dict[str, Any]) -> float:
        """Score by comparing extracted answer to reference."""
        predicted = self.extract_answer(response)
        if predicted is None:
            return 0.0
        
        # Reference answer is in "answer" field, often with #### prefix
        ref_answer = reference.get("answer", "")
        if isinstance(ref_answer, str):
            # Extract number from reference if it has #### prefix
            match = re.search(r"####\s*(\d+\.?\d*)", ref_answer)
            if match:
                ref_value = float(match.group(1))
            else:
                try:
                    ref_value = float(ref_answer)
                except ValueError:
                    return 0.0
        else:
            ref_value = float(ref_answer)
        
        return 1.0 if abs(predicted - ref_value) < 1e-6 else 0.0
    
    def is_correct(self, prompt: str, response: str, reference: dict[str, Any]) -> bool:
        """Check if answer matches exactly."""
        return self.score(prompt, response, reference) == 1.0
