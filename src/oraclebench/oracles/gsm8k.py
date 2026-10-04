"""GSM8K scorer: exact numeric match on final answer."""

import re
from typing import Any

from .base import BaseScorer


class GSM8KScorerV1(BaseScorer):
    """Original scorer (copied from flipgate v1), kept to reproduce the first OracleBench numbers.

    Weaknesses: ignores \\boxed{} answers and reads "1,234" as 234.
    """

    version = "1"
    
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


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def _to_num(text: str) -> float | None:
    try:
        return float(text.replace(",", "").replace("$", "").strip().rstrip("."))
    except ValueError:
        return None


class GSM8KScorer(GSM8KScorerV1):
    """Robust scorer (v2, same logic as flipgate's GSM8KScorer v2): last \\boxed{}, '####',
    'answer is', then the last number; thousands separators handled."""

    version = "2"

    def extract_answer(self, response: str) -> float | None:
        boxed = _BOXED.findall(response)
        if boxed:
            nums = _NUM.findall(boxed[-1].replace("\\,", ""))
            if nums:
                return _to_num(nums[-1])
        m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", response)
        if m:
            return _to_num(m.group(1))
        m = re.search(r"answer is[^\d\-]{0,20}(-?\d[\d,]*\.?\d*)", response, re.IGNORECASE)
        if m:
            return _to_num(m.group(1))
        nums = _NUM.findall(response)
        return _to_num(nums[-1]) if nums else None

    def score(self, prompt: str, response: str, reference: dict[str, Any]) -> float:
        predicted = self.extract_answer(response)
        if predicted is None:
            return 0.0
        ref = reference.get("answer", "")
        if isinstance(ref, str):
            m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", ref)
            gold = _to_num(m.group(1)) if m else _to_num(ref)
        else:
            gold = float(ref)
        if gold is None:
            return 0.0
        return 1.0 if abs(predicted - gold) < 1e-6 else 0.0
