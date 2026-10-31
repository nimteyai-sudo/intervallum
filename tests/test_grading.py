from __future__ import annotations

from pathlib import Path

from intervallum_ml.content import load_content
from intervallum_ml.grading import grade_answer
from intervallum_ml.types import Task, TaskMechanic

ROOT = Path(__file__).parents[1]
BANK = ROOT / "content" / "fipi_physics_2026.json"


def by_type(bank, answer_type: str):
    return next(t for t in bank.tasks.values() if t.answer_type == answer_type)


def test_single_digit_grading():
    bank = load_content(BANK)
    task = by_type(bank, "single_digit")
    correct = task.answer_digits[0]
    wrong = 1 if correct != 1 else 2
    assert grade_answer(task, str(correct)).raw_score == 1.0
    assert grade_answer(task, f" {correct} ").raw_score == 1.0
    assert grade_answer(task, str(wrong)).raw_score == 0.0
    assert grade_answer(task, f"{correct}{wrong}").raw_score == 0.0
    assert grade_answer(task, "").raw_score == 0.0


def test_multi_digit_grading_is_set_based():
    bank = load_content(BANK)
    task = by_type(bank, "multi_digit")
    digits = list(task.answer_digits)
    correct = "".join(str(d) for d in digits)
    reversed_answer = "".join(str(d) for d in reversed(digits))
    assert grade_answer(task, correct).raw_score == 1.0
    assert grade_answer(task, reversed_answer).raw_score == 1.0
    wrong = "".join(str(d) for d in digits[:-1])
    assert grade_answer(task, wrong).raw_score == 0.0
    partial = grade_answer(task, wrong, partial_multi=True)
    assert 0.0 <= partial.raw_score < 1.0


def test_code_answers_are_order_and_repetition_sensitive():
    bank = load_content(BANK)
    task = next(
        t for t in bank.tasks.values()
        if t.answer_type == "code" and t.answer_raw != t.answer_raw[::-1]
    )
    correct = task.answer_raw
    assert grade_answer(task, correct).raw_score == 1.0
    assert grade_answer(task, correct[::-1]).raw_score == 0.0
    assert grade_answer(task, correct[:1]).raw_score == 0.0


def test_numeric_tolerance():
    bank = load_content(BANK)
    task = by_type(bank, "numeric")
    value = task.answer_value
    assert value is not None
    assert grade_answer(task, f"{value:.6f}").raw_score == 1.0
    assert grade_answer(task, f"{value:.6f}".replace(".", ",")).raw_score == 1.0
    assert grade_answer(task, f"{value:.6f} м/с").raw_score == 1.0
    tolerance = max(1e-6, 0.01 * abs(value))
    assert grade_answer(task, str(value + tolerance * 0.5)).raw_score == 1.0
    assert grade_answer(task, str(value + tolerance * 5.0)).raw_score == 0.0


def test_measurement_grading_ignores_spaces_and_separators():
    bank = load_content(BANK)
    task = by_type(bank, "measurement")
    correct = task.answer_raw
    assert grade_answer(task, correct).raw_score == 1.0
    assert grade_answer(task, correct.replace(",", " ")).raw_score == 1.0
    assert grade_answer(task, correct + "9").raw_score == 0.0


def test_free_response_requires_review():
    task = Task(
        1,
        TaskMechanic.FREE_RESPONSE,
        (1,),
        answer_raw="сила тока",
        answer_type="text",
    )
    ok = grade_answer(task, "Сила тока")
    assert ok.raw_score == 1.0
    bad = grade_answer(task, "напряжение")
    assert bad.raw_score == 0.0
    assert bad.needs_manual_review


def test_every_real_task_grades_its_own_answer_as_correct():
    bank = load_content(BANK)
    failures = []
    for task in bank.tasks.values():
        result = grade_answer(task, task.answer_raw)
        if result.raw_score != 1.0:
            failures.append((task.id, task.answer_type, task.answer_raw, result.raw_score))
    assert not failures, failures[:20]


def test_every_real_task_rejects_a_wrong_answer():
    bank = load_content(BANK)
    failures = []
    for task in bank.tasks.values():
        kind = task.answer_type
        if kind == "single_digit":
            correct = task.answer_digits[0]
            wrong = str((correct % 5) + 1)
        elif kind == "multi_digit":
            present = set(task.answer_digits)
            missing = next((d for d in range(1, 6) if d not in present), None)
            if missing is None:
                continue
            wrong = "".join(str(d) for d in task.answer_digits) + str(missing)
        elif kind == "code":
            wrong = task.answer_raw + "1"
        elif kind == "numeric" and task.answer_value is not None:
            wrong = str(task.answer_value + max(0.1, abs(task.answer_value) * 0.2))
        elif kind == "measurement":
            wrong = task.answer_raw + "9"
        else:
            continue
        result = grade_answer(task, wrong)
        if result.raw_score != 0.0:
            failures.append((task.id, kind, task.answer_raw, wrong, result.raw_score))
    assert not failures, failures[:20]
