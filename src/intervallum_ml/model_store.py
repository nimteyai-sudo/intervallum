"""Versioned JSON model artefacts.

The production architecture deliberately avoids a heavy model registry at the
start.  A nightly job writes a small JSON artefact to object storage; the
TypeScript backend reads the active version and hot-reloads it.  This module
implements the artefact contract and a safe atomic write.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .types import CalibrationReport


@dataclass(frozen=True)
class ModelArtifact:
    version: str
    created_at: str
    tau: float
    gamma: float
    b: Mapping[int, float]
    a: Mapping[int, float]
    metrics: Mapping[str, float] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_report(
        cls,
        report: CalibrationReport,
        *,
        version: str | None = None,
        notes: str = "",
    ) -> "ModelArtifact":
        version = version or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return cls(
            version=version,
            created_at=datetime.now(timezone.utc).isoformat(),
            tau=report.tau,
            gamma=report.gamma,
            b={int(k): float(v) for k, v in report.b.items()},
            a={int(k): float(v) for k, v in report.a.items()},
            metrics={
                "brier": report.brier,
                "ece": report.ece,
                "loss": report.loss,
                "n_attempts": float(report.n_attempts),
                "n_items": float(report.n_items),
            },
            notes=notes,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def save_model(path: str | Path, artifact: ModelArtifact) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_model(path: str | Path) -> ModelArtifact:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ModelArtifact(
        version=data["version"],
        created_at=data.get("created_at", ""),
        tau=float(data["tau"]),
        gamma=float(data["gamma"]),
        b={int(k): float(v) for k, v in data.get("b", {}).items()},
        a={int(k): float(v) for k, v in data.get("a", {}).items()},
        metrics={k: float(v) for k, v in data.get("metrics", {}).items()},
        notes=data.get("notes", ""),
    )
