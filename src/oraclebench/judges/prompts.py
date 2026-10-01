"""Frozen judge prompt templates (v1). Edit only by adding v2+."""

import json


def _ref_summary(reference: dict) -> str:
    if "answer" in reference and reference["answer"]:
        return f"Expected answer: {reference['answer']}"
    if "instruction_id_list" in reference:
        return ("Must satisfy: " + "; ".join(reference["instruction_id_list"]))
    return "Must cite a real FAR/DFARS clause number from the official registry."


def render_pointwise(prompt: str, answer: str, reference: dict) -> str:
    return (
        "You are a strict grader. Judge ONLY whether the answer satisfies every "
        "explicit requirement in the question. Reply with exactly one word: CORRECT or WRONG.\n"
        f"\nQuestion:\n{prompt}\n"
        f"\nCandidate answer:\n{answer}\n"
        f"\nReference (ground truth for comparison):\n{_ref_summary(reference)}\n"
        "\nVerdict (CORRECT or WRONG):"
    )


def render_pairwise(prompt: str, answer_a: str, answer_b: str, reference: dict) -> str:
    return (
        "You are a strict grader. Two candidate answers (A and B) respond to the same "
        "question. Judge which one better satisfies every explicit requirement. "
        "Reply with exactly one word: A or B.\n"
        f"\nQuestion:\n{prompt}\n"
        f"\nAnswer A:\n{answer_a}\n"
        f"\nAnswer B:\n{answer_b}\n"
        f"\nReference (ground truth for comparison):\n{_ref_summary(reference)}\n"
        "\nVerdict (A or B):"
    )
