"""Serialisation helpers for PostgreSQL/JSONB state.

The online state is small (one row per user x knowledge unit).  A production
adapter can store this as JSONB or as normal columns; these helpers make the
Python reference implementation portable and testable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

from .types import MemoryState, UserKUState


def _dt(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def memory_to_dict(memory: MemoryState | None) -> dict | None:
    if memory is None:
        return None
    return {
        "stability": memory.stability,
        "difficulty": memory.difficulty,
        "state": memory.state,
        "step": memory.step,
        "last_review_at": _dt(memory.last_review_at),
        "due_at": _dt(memory.due_at),
    }


def memory_from_dict(data: Mapping | None) -> MemoryState | None:
    if not data:
        return None
    return MemoryState(
        stability=float(data.get("stability", 0.0)),
        difficulty=float(data.get("difficulty", 5.0)),
        state=int(data.get("state", 1)),
        step=data.get("step"),
        last_review_at=_parse(data.get("last_review_at")),
        due_at=_parse(data.get("due_at")),
    )


def state_to_dict(state: UserKUState) -> dict:
    return {
        "ku_id": state.ku_id,
        "theta": state.theta,
        "sigma": state.sigma,
        "memory": memory_to_dict(state.memory),
        "attempts_count": state.attempts_count,
        "last_seen_at": _dt(state.last_seen_at),
    }


def state_from_dict(data: Mapping) -> UserKUState:
    return UserKUState(
        ku_id=int(data["ku_id"]),
        theta=float(data.get("theta", 0.0)),
        sigma=float(data.get("sigma", 1.0)),
        memory=memory_from_dict(data.get("memory")),
        attempts_count=int(data.get("attempts_count", 0)),
        last_seen_at=_parse(data.get("last_seen_at")),
    )


def states_to_dict(states: Mapping[int, UserKUState]) -> dict[str, dict]:
    return {str(k): state_to_dict(v) for k, v in states.items()}


def states_from_dict(data: Mapping[str, Mapping]) -> dict[int, UserKUState]:
    return {int(k): state_from_dict(v) for k, v in data.items()}
