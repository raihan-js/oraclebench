"""IFEval scorer: rule-based instruction-following checks.

Implements checkers for the 25 instruction types in google/IFEval,
following the semantics of the official evaluation (Zhou et al. 2023).
This is an independent reimplementation validated on hand-made cases;
it vendors no code, only the published rules.

Reference format (matches the IFEval dataset):
    reference = {
        "instruction_id_list": ["punctuation:no_comma", ...],
        "kwargs": [{}, ...],  # one kwargs dict per instruction
    }
"""

import json
import re
from typing import Any

from .base import BaseScorer

# Map ISO 639-1 codes used in IFEval to langdetect codes (same standard).
# langdetect returns ISO 639-1; compare directly.


class IFEvalScorer(BaseScorer):
    """Score IFEval responses using rule-based checks."""

    def score(self, prompt: str, response: str, reference: dict[str, Any]) -> float:
        """Fraction of instruction constraints satisfied."""
        instructions = reference.get("instruction_id_list", [])
        kwargs_list = reference.get("kwargs", [])

        if not instructions:
            return 1.0

        passed = 0
        for i, instruction_id in enumerate(instructions):
            kwargs = kwargs_list[i] if i < len(kwargs_list) else {}
            if self._check(instruction_id, prompt, response, kwargs):
                passed += 1

        return passed / len(instructions)

    def is_correct(self, prompt: str, response: str, reference: dict[str, Any]) -> bool:
        """All instruction constraints satisfied."""
        return self.score(prompt, response, reference) == 1.0

    # -- dispatcher ---------------------------------------------------------

    def _check(self, instruction_id: str, prompt: str, response: str, kwargs: dict) -> bool:
        handlers = {
            "punctuation:no_comma": self._no_comma,
            "length_constraints:number_words": self._number_words,
            "length_constraints:number_sentences": self._number_sentences,
            "length_constraints:number_paragraphs": self._number_paragraphs,
            "length_constraints:nth_paragraph_first_word": self._nth_paragraph_first_word,
            "keywords:forbidden_words": self._forbidden_words,
            "keywords:existence": self._keyword_existence,
            "keywords:frequency": self._keyword_frequency,
            "keywords:letter_frequency": self._letter_frequency,
            "detectable_format:number_highlighted_sections": self._number_highlights,
            "detectable_format:number_bullet_lists": self._number_bullets,
            "detectable_format:title": self._title,
            "detectable_format:json_format": self._json_format,
            "detectable_format:multiple_sections": self._multiple_sections,
            "detectable_format:constrained_response": self._constrained_response,
            "detectable_content:number_placeholders": self._number_placeholders,
            "detectable_content:postscript": self._postscript,
            "language:response_language": self._response_language,
            "startend:quotation": self._quotation,
            "startend:end_checker": self._end_checker,
            "change_case:english_lowercase": self._english_lowercase,
            "change_case:english_capital": self._english_capital,
            "change_case:capital_word_frequency": self._capital_word_frequency,
            "combination:repeat_prompt": self._repeat_prompt,
            "combination:two_responses": self._two_responses,
        }
        handler = handlers.get(instruction_id)
        if handler is None:
            return True  # unknown instruction type: do not fail
        try:
            # constrained_response needs the prompt to find the options
            if instruction_id == "detectable_format:constrained_response":
                return self._constrained_response(prompt, response, kwargs)
            return handler(response, kwargs)
        except Exception:
            return False

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _compare(value: float, relation: str, target: float) -> bool:
        relation = (relation or "").lower()
        if relation in ("at least", "more than", ">="):
            return value >= target
        if relation in ("at most", "less than", "fewer than", "<="):
            return value <= target
        if relation in ("exactly", "equal to", "=="):
            return value == target
        if relation in ("more than", ">"):
            return value > target
        if relation in ("less than", "<"):
            return value < target
        return value == target

    # -- punctuation --------------------------------------------------------

    @staticmethod
    def _no_comma(response: str, kwargs: dict) -> bool:
        return "," not in response

    # -- length constraints -------------------------------------------------

    def _number_words(self, response: str, kwargs: dict) -> bool:
        count = len(response.split())
        return self._compare(count, kwargs.get("relation", ""), kwargs.get("num_words", 0))

    def _number_sentences(self, response: str, kwargs: dict) -> bool:
        sentences = [s for s in re.split(r"[.!?]+", response) if s.strip()]
        return self._compare(len(sentences), kwargs.get("relation", ""), kwargs.get("num_sentences", 0))

    def _number_paragraphs(self, response: str, kwargs: dict) -> bool:
        paragraphs = [p for p in response.split("\n\n") if p.strip()]
        return self._compare(len(paragraphs), kwargs.get("relation", ""), kwargs.get("num_paragraphs", 0))

    def _nth_paragraph_first_word(self, response: str, kwargs: dict) -> bool:
        paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
        nth = kwargs.get("nth_paragraph", 1)
        first_word = kwargs.get("first_word", "")
        if len(paragraphs) < nth:
            return False
        paragraph = paragraphs[nth - 1]
        tokens = paragraph.split()
        if not tokens:
            return False
        actual = tokens[0].lstrip("\"'“‘([{").rstrip(",.:;!?\"'”’")
        return actual.lower() == first_word.lower()

    # -- keywords -----------------------------------------------------------

    @staticmethod
    def _forbidden_words(response: str, kwargs: dict) -> bool:
        lowered = response.lower()
        for word in kwargs.get("forbidden_words", []):
            if re.search(r"\b" + re.escape(word.lower()) + r"\b", lowered):
                return False
        return True

    @staticmethod
    def _keyword_existence(response: str, kwargs: dict) -> bool:
        lowered = response.lower()
        return all(
            re.search(r"\b" + re.escape(kw.lower()) + r"\b", lowered)
            for kw in kwargs.get("keywords", [])
        )

    def _keyword_frequency(self, response: str, kwargs: dict) -> bool:
        keyword = kwargs.get("keyword", "").lower()
        count = len(re.findall(r"\b" + re.escape(keyword) + r"\b", response.lower()))
        return self._compare(count, kwargs.get("relation", ""), kwargs.get("frequency", 0))

    def _letter_frequency(self, response: str, kwargs: dict) -> bool:
        letter = kwargs.get("letter", "")
        count = response.count(letter)
        return self._compare(count, kwargs.get("let_relation", ""), kwargs.get("let_frequency", 0))

    # -- detectable format --------------------------------------------------

    @staticmethod
    def _number_highlights(response: str, kwargs: dict) -> bool:
        # Markdown highlights: *text* (single asterisks, not bullets/bold)
        highlights = re.findall(r"(?<!\*)\*([^*\n]+)\*(?!\*)", response)
        return len(highlights) >= kwargs.get("num_highlights", 0)

    @staticmethod
    def _number_bullets(response: str, kwargs: dict) -> bool:
        bullets = re.findall(r"(?m)^\s*\*\s+\S", response)
        return len(bullets) == kwargs.get("num_bullets", 0)

    @staticmethod
    def _title(response: str, kwargs: dict) -> bool:
        return re.search(r"<<.+?>>", response, re.DOTALL) is not None

    @staticmethod
    def _json_format(response: str, kwargs: dict) -> bool:
        text = response.strip()
        # Strip markdown fences (```json ... ``` or ``` ... ```)
        fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    @staticmethod
    def _multiple_sections(response: str, kwargs: dict) -> bool:
        splitter = kwargs.get("section_spliter", "")
        num_sections = kwargs.get("num_sections", 0)
        if not splitter:
            return False
        return response.count(splitter) == num_sections

    @staticmethod
    def _constrained_response(prompt: str, response: str, kwargs: dict) -> bool:
        # Options live in the prompt ("Choose from the following: ('A', 'B', ...)").
        # Extract quoted phrases from the option-list portion and require
        # the response to contain at least one of them exactly.
        options = kwargs.get("options")
        if not options:
            marker = re.search(r"following:?\s*(.*)", prompt, re.DOTALL | re.IGNORECASE)
            option_text = marker.group(1) if marker else prompt
            # Straight and curly quotes; options are full sentences.
            options = re.findall(r"['\"“”]([^'\"“”]{8,}?)['\"“”]", option_text)
        if not options:
            return False
        return sum(1 for o in options if o in response) >= 1

    # -- detectable content -------------------------------------------------

    @staticmethod
    def _number_placeholders(response: str, kwargs: dict) -> bool:
        placeholders = re.findall(r"\[[^\[\]]+\]", response)
        return len(placeholders) >= kwargs.get("num_placeholders", 0)

    @staticmethod
    def _postscript(response: str, kwargs: dict) -> bool:
        marker = kwargs.get("postscript_marker", "P.S.")
        return marker in response

    # -- language -----------------------------------------------------------

    @staticmethod
    def _response_language(response: str, kwargs: dict) -> bool:
        from langdetect import detect, LangDetectException

        target = kwargs.get("language", "")
        if not target or not response.strip():
            return False
        try:
            return detect(response) == target
        except LangDetectException:
            return False

    # -- start / end --------------------------------------------------------

    @staticmethod
    def _quotation(response: str, kwargs: dict) -> bool:
        text = response.strip()
        return len(text) >= 2 and text.startswith('"') and text.endswith('"')

    @staticmethod
    def _end_checker(response: str, kwargs: dict) -> bool:
        end_phrase = kwargs.get("end_phrase", "")
        return response.rstrip().endswith(end_phrase)

    # -- change case --------------------------------------------------------

    @staticmethod
    def _english_lowercase(response: str, kwargs: dict) -> bool:
        return re.search(r"[A-Z]", response) is None

    @staticmethod
    def _english_capital(response: str, kwargs: dict) -> bool:
        return re.search(r"[a-z]", response) is None

    def _capital_word_frequency(self, response: str, kwargs: dict) -> bool:
        count = sum(
            1 for tok in re.findall(r"[A-Za-z]+", response)
            if tok.isupper()
        )
        return self._compare(
            count, kwargs.get("capital_relation", ""), kwargs.get("capital_frequency", 0)
        )

    # -- combination --------------------------------------------------------

    @staticmethod
    def _repeat_prompt(response: str, kwargs: dict) -> bool:
        prompt_to_repeat = kwargs.get("prompt_to_repeat", "")
        return response.lstrip().startswith(prompt_to_repeat)

    @staticmethod
    def _two_responses(response: str, kwargs: dict) -> bool:
        parts = [p for p in response.split("******") if p.strip()]
        return len(parts) == 2
