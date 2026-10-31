"""Answer grading for the imported EGE task types.

The ML engine consumes a ``raw_score`` in [0, 1].  This module turns a student's
answer into that score in a deterministic, testable way.

Important distinctions:

* ``multi_digit`` is a set of selected options (order-independent);
* ``code`` is a two/three-digit EGE code (order and repetition are meaningful);
* ``measurement`` is a value+uncertainty string such as ``450,012,5``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .types import Task


@dataclass(frozen=True)
class GradeResult:
    task_id: int
    raw_score: float
    correct: bool
    normalized_user: str
    normalized_correct: str
    method: str
    needs_manual_review: bool = False


def normalize_text(value: str | int | float) -> str:
    text = str(value).strip().lower()
    text = text.replace("−", "-").replace("–", "-")
    text = text.replace("ё", "е")
    text = re.sub(r"\s+", "", text)
    return text


def _digits(value: str) -> list[int]:
    return [int(ch) for ch in re.findall(r"[1-5]", value)]


def _all_digits(value: str) -> str:
    return re.sub(r"[^0-9]", "", value)


def _parse_number(value: str | int | float) -> float | None:
    text = normalize_text(value).replace(",", ".")
    # Accept a number with a unit suffix, e.g. "3.5 м/с".
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match:
        text = match.group(0)
    try:
        return float(text)
    except ValueError:
        return None


def grade_answer(
    task: Task,
    answer: str | int | float,
    *,
    numeric_abs_tol: float = 1e-6,
    numeric_rel_tol: float = 0.01,
    partial_multi: bool = False,
) -> GradeResult:
    user_norm = normalize_text(answer)
    correct_norm = normalize_text(task.answer_raw)
    answer_type = task.answer_type or "text"

    if answer_type == "single_digit":
        user_digits = _digits(user_norm)
        correct_digits = _digits(correct_norm)
        correct = (
            len(user_digits) == 1
            and bool(correct_digits)
            and user_digits[0] == correct_digits[0]
        )
        return GradeResult(
            task.id, 1.0 if correct else 0.0, correct, user_norm, correct_norm, "single_digit"
        )

    if answer_type == "multi_digit":
        user_set = set(_digits(user_norm))
        correct_set = set(task.answer_digits) if task.answer_digits else set(_digits(correct_norm))
        if not correct_set:
            return GradeResult(
                task.id, 0.0, False, user_norm, correct_norm, "multi_digit", True
            )
        if user_set == correct_set:
            score = 1.0
        elif partial_multi:
            union = user_set | correct_set
            score = max(0.0, 1.0 - len(user_set ^ correct_set) / len(union))
        else:
            score = 0.0
        return GradeResult(
            task.id, score, user_set == correct_set, user_norm, correct_norm, "multi_digit"
        )

    if answer_type == "code":
        user_digits = _digits(user_norm)
        correct_digits = list(task.answer_digits) if task.answer_digits else _digits(correct_norm)
        correct = user_digits == correct_digits
        return GradeResult(
            task.id, 1.0 if correct else 0.0, correct, user_norm, correct_norm, "code"
        )

    if answer_type == "numeric":
        correct_value = task.answer_value
        user_value = _parse_number(answer)
        if correct_value is None or user_value is None:
            return GradeResult(
                task.id, 0.0, False, user_norm, correct_norm, "numeric", True
            )
        tolerance = max(numeric_abs_tol, numeric_rel_tol * abs(correct_value))
        correct = abs(user_value - correct_value) <= tolerance
        return GradeResult(
            task.id, 1.0 if correct else 0.0, correct, user_norm, correct_norm, "numeric"
        )

    if answer_type == "measurement":
        user_measurement = _all_digits(user_norm)
        correct_measurement = _all_digits(correct_norm)
        correct = bool(user_measurement) and user_measurement == correct_measurement
        return GradeResult(
            task.id, 1.0 if correct else 0.0, correct, user_norm, correct_norm, "measurement"
        )

    # Free response / unknown type: exact match only; flag for manual review.
    correct = bool(user_norm) and user_norm == correct_norm
    return GradeResult(
        task.id,
        1.0 if correct else 0.0,
        correct,
        user_norm,
        correct_norm,
        answer_type or "text",
        needs_manual_review=not correct,
    )
