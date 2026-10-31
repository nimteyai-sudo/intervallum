"""Content-bank loading and validation.

The ML engine is only as good as its content.  This module makes content errors
visible at load time: missing answers, dangling knowledge-unit references,
missing media for image-dependent tasks, duplicate IDs and empty texts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .types import KnowledgeUnit, KUType, Task, TaskMechanic


@dataclass(frozen=True)
class ContentIssue:
    severity: str  # "error" | "warning"
    code: str
    message: str
    task_id: int | None = None
    ku_id: int | None = None


@dataclass(frozen=True)
class ContentValidationReport:
    task_count: int
    ku_count: int
    media_tasks: int
    issues: tuple[ContentIssue, ...]

    @property
    def errors(self) -> tuple[ContentIssue, ...]:
        return tuple(i for i in self.issues if i.severity == "error")

    @property
    def warnings(self) -> tuple[ContentIssue, ...]:
        return tuple(i for i in self.issues if i.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        if self.errors:
            messages = "; ".join(f"{i.code}: {i.message}" for i in self.errors[:20])
            raise ValueError(f"Invalid content bank: {messages}")


@dataclass(frozen=True)
class ContentBank:
    knowledge_units: Mapping[int, KnowledgeUnit]
    tasks: Mapping[int, Task]
    meta: Mapping
    path: Path
    issues: tuple[ContentIssue, ...] = field(default_factory=tuple)

    def validation_report(self) -> ContentValidationReport:
        return validate_content(self)


def _issue(
    severity: str,
    code: str,
    message: str,
    *,
    task_id: int | None = None,
    ku_id: int | None = None,
) -> ContentIssue:
    return ContentIssue(severity, code, message, task_id, ku_id)


def validate_content(bank: ContentBank) -> ContentValidationReport:
    issues: list[ContentIssue] = []
    tasks = bank.tasks
    kus = bank.knowledge_units
    if len(tasks) != len(set(tasks)):
        issues.append(_issue("error", "duplicate_task_ids", "Task IDs are not unique"))
    if len(kus) != len(set(kus)):
        issues.append(_issue("error", "duplicate_ku_ids", "Knowledge-unit IDs are not unique"))

    media_tasks = 0
    for task in tasks.values():
        if not task.text.strip():
            issues.append(_issue("error", "empty_text", "Task has empty text", task_id=task.id))
        if not task.answer_raw.strip():
            issues.append(_issue("error", "missing_answer", "Task has no answer", task_id=task.id))
        if not task.ku_ids:
            issues.append(_issue("error", "missing_ku", "Task has no knowledge units", task_id=task.id))
        for ku_id in task.ku_ids:
            if ku_id not in kus:
                issues.append(
                    _issue(
                        "error",
                        "dangling_ku",
                        f"Task references unknown KU {ku_id}",
                        task_id=task.id,
                        ku_id=ku_id,
                    )
                )
        if task.answer_type == "single_digit" and len(task.answer_digits) != 1:
            issues.append(
                _issue("error", "bad_single_answer", "Single-choice answer must have one digit", task_id=task.id)
            )
        if task.answer_type == "multi_digit" and len(set(task.answer_digits)) < 2:
            issues.append(
                _issue("error", "bad_multi_answer", "Multi-choice answer must have at least two unique digits", task_id=task.id)
            )
        if task.answer_type == "code" and len(task.answer_digits) < 2:
            issues.append(
                _issue("error", "bad_code_answer", "Code answer must have at least two digits", task_id=task.id)
            )
        if task.answer_type == "numeric" and task.answer_value is None:
            issues.append(
                _issue("error", "bad_numeric_answer", "Numeric answer has no parsed value", task_id=task.id)
            )
        if task.answer_type == "measurement" and not task.answer_raw:
            issues.append(
                _issue("error", "bad_measurement_answer", "Measurement answer is empty", task_id=task.id)
            )
        if task.mechanic == TaskMechanic.MCQ_SINGLE and task.c_guess <= 0:
            issues.append(
                _issue("warning", "single_choice_zero_guess", "Single choice has zero guessing floor", task_id=task.id)
            )
        if task.media:
            media_tasks += 1
            for media in task.media:
                media_path = Path(media)
                if not media_path.is_absolute():
                    media_path = bank.path.parent.parent / media_path
                if not media_path.exists():
                    issues.append(
                        _issue(
                            "error",
                            "missing_media",
                            f"Media file not found: {media}",
                            task_id=task.id,
                        )
                    )
        elif any(tag == "media_required:true" for tag in task.tags):
            issues.append(
                _issue(
                    "warning",
                    "media_required_but_empty",
                    "Task requires a figure but has no media",
                    task_id=task.id,
                )
            )

    # Orphan KUs are allowed (they may be introduced later), but report them.
    used = {kid for t in tasks.values() for kid in t.ku_ids}
    for ku_id in kus:
        if ku_id not in used:
            issues.append(
                _issue("warning", "orphan_ku", f"KU {ku_id} is not used by any task", ku_id=ku_id)
            )

    return ContentValidationReport(
        task_count=len(tasks),
        ku_count=len(kus),
        media_tasks=media_tasks,
        issues=tuple(issues),
    )


def load_content(path: str | Path, *, validate: bool = True) -> ContentBank:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    kus: dict[int, KnowledgeUnit] = {}
    for row in data.get("knowledge_units", []):
        ku_id = int(row["id"])
        if ku_id in kus:
            raise ValueError(f"Duplicate knowledge-unit id: {ku_id}")
        ku = KnowledgeUnit(
            id=ku_id,
            code=str(row["code"]),
            name=str(row["name"]),
            ku_type=KUType(row.get("ku_type", "hybrid")),
            topic_id=int(row.get("topic_id", 0)),
            section_id=int(row.get("section_id", 0)),
            ege_weight=float(row.get("ege_weight", 1.0)),
            parent_ids=tuple(int(x) for x in row.get("parent_ids", [])),
            neighbor_ids=tuple(int(x) for x in row.get("neighbor_ids", [])),
        )
        kus[ku.id] = ku

    tasks: dict[int, Task] = {}
    for row in data.get("tasks", []):
        task_id = int(row["id"])
        if task_id in tasks:
            raise ValueError(f"Duplicate task id: {task_id}")
        task = Task(
            id=task_id,
            mechanic=TaskMechanic(row["mechanic"]),
            ku_ids=tuple(int(x) for x in row.get("ku_ids", [])),
            ku_weights=tuple(float(x) for x in row.get("ku_weights", [])),
            c_guess=float(row.get("c_guess", 0.0)),
            t_expected_sec=float(row.get("t_expected_sec", 60.0)),
            b_difficulty=float(row.get("b_difficulty", 0.0)),
            a_discrimination=float(row.get("a_discrimination", 1.0)),
            expert_difficulty=int(row.get("expert_difficulty", 3)),
            is_quarantined=bool(row.get("is_quarantined", False)),
            topic_id=row.get("topic_id"),
            section_id=row.get("section_id"),
            text=str(row.get("text", "")),
            source=str(row.get("source", "")),
            version=int(row.get("version", 1)),
            media=tuple(str(x) for x in row.get("media", [])),
            tags=tuple(str(x) for x in row.get("tags", [])),
            answer_raw=str(row.get("answer_raw", "")),
            answer_type=str(row.get("answer_type", "")),
            answer_value=float(row["answer_value"]) if row.get("answer_value") is not None else None,
            answer_digits=tuple(int(x) for x in row.get("answer_digits", [])),
        )
        tasks[task.id] = task

    bank = ContentBank(
        knowledge_units=kus,
        tasks=tasks,
        meta=dict(data.get("meta", {})),
        path=path,
        issues=(),
    )
    if validate:
        report = validate_content(bank)
        report.raise_if_invalid()
    return bank


def load_engine(path: str | Path, config=None):
    from .engine import IntervallumEngine

    bank = load_content(path)
    return IntervallumEngine(bank.knowledge_units, bank.tasks, config), bank
