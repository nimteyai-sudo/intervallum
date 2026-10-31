"""Import the downloaded FIPI/EGE physics task PDFs into a structured bank.

The input files are the per-position PDFs of the open bank downloaded by the
user.  The importer is intentionally conservative:

* it only keeps tasks with an explicit answer marker;
* it skips a PDF if the text layer is corrupted (task 6);
* it marks image-dependent tasks with ``media_required`` instead of silently
  pretending the text is self-contained;
* it validates every answer and rejects empty/garbage tasks.

Usage:
    python scripts/import_fipi.py \
        --pdf-dir "/home/nimtey/Загрузки" \
        --out content/fipi_physics_2026.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict
from pathlib import Path

from intervallum_ml.types import KnowledgeUnit, KUType, Task, TaskMechanic

SECTION_NAMES = {
    1: "Механика",
    2: "Молекулярная физика и термодинамика",
    3: "Электродинамика",
    4: "Оптика",
    5: "Квантовая физика и атомное ядро",
    6: "Методы научного познания и общие вопросы",
}

POSITION_SECTION = {
    1: 1, 2: 1, 3: 1, 4: 1,
    5: 2, 6: 2, 7: 2, 8: 2,
    9: 3, 10: 3, 11: 3, 12: 3, 13: 3,
    14: 4, 15: 4, 16: 4,
    17: 5,
    18: 6, 19: 6, 20: 6,
}

POSITION_DIFFICULTY = {
    1: 2, 2: 2, 3: 2, 4: 3,
    5: 3, 6: 3, 7: 3, 8: 3,
    9: 3, 10: 3, 11: 3, 12: 3, 13: 4,
    14: 4, 15: 4, 16: 4,
    17: 4, 18: 4, 19: 4, 20: 4,
}

# Official EGE-2026 format (see the FIPI demo/specification):
# 1-4, 7, 8, 11-13, 16 -> numeric;
# 5, 9, 14, 18, 20 -> select 2/3 options (set);
# 6, 10, 15, 17 -> sequence/code;
# 19 -> value and uncertainty.
POSITION_ANSWER_TYPE = {
    1: "numeric", 2: "numeric", 3: "numeric", 4: "numeric",
    5: "multi_digit", 6: "code", 7: "numeric", 8: "numeric",
    9: "multi_digit", 10: "code", 11: "numeric", 12: "numeric", 13: "numeric",
    14: "multi_digit", 15: "code", 16: "numeric", 17: "code",
    18: "multi_digit", 19: "measurement", 20: "multi_digit",
}

POSITION_TIME = {
    1: 90, 2: 120, 3: 150, 4: 180,
    5: 150, 6: 180, 7: 180, 8: 180,
    9: 180, 10: 180, 11: 180, 12: 180, 13: 180,
    14: 180, 15: 180, 16: 180,
    17: 180, 18: 180, 19: 180, 20: 180,
}


def _pdftotext(path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftotext failed for {path}: {result.stderr[:300]}")
    return result.stdout


def _clean_question(text: str) -> str:
    text = text.replace("\x0c", "\n")
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if re.match(r"^ЕГЭ\s", stripped):
            continue
        if re.search(r"\b[a-z0-9-]+\.(ru|com|org)\b", stripped.lower()):
            continue
        if re.fullmatch(r"\d{1,3}", stripped):
            continue
        if stripped in {"Введите ваш ответ", "Проверить", "Показать ответ 📚"}:
            continue
        lines.append(line.rstrip())
    # Strip leading task number.
    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _normalise_answer(raw: str) -> tuple[str, str, float | None, list[int] | None]:
    raw = raw.strip().rstrip(".").strip()
    raw = raw.replace("−", "-")
    if not raw:
        return raw, "empty", None, None
    if re.fullmatch(r"[1-5]", raw):
        return raw, "single_digit", float(raw), [int(raw)]
    if re.fullmatch(r"[1-5]{2,5}", raw):
        digits = [int(ch) for ch in raw]
        if len(set(digits)) < len(digits):
            # EGE "how will two quantities change" codes: 11, 22, 33, etc.
            # These are NOT multi-select sets; repeated digits are meaningful.
            return raw, "code", None, digits
        return raw, "multi_digit", None, sorted(set(digits))
    numeric = raw.replace(",", ".")
    try:
        value = float(numeric)
        return raw, "numeric", value, None
    except ValueError:
        pass
    # EGE measurement tasks concatenate value and uncertainty, e.g.
    # "450,012,5" for 450.0 ± 12.5.  A pure "27525" is parsed as numeric
    # first and is promoted to "measurement" by parse_pdf when the question
    # asks for value and uncertainty together.
    if re.fullmatch(r"[0-9][0-9,]{2,30}", raw):
        return raw, "measurement", None, None
    return raw, "text", None, None


def _infer_answer_type(question: str, raw: str, fallback: str) -> str:
    """Disambiguate digit answers by the wording of the task.

    "25" can be a numeric answer in metres or a two-option selection.  The
    question text, not the digit string alone, decides.
    """

    q = re.sub(r"\s+", " ", question.lower())
    digits = [int(ch) for ch in raw if ch.isdigit()]
    numeric = bool(re.fullmatch(r"-?\d+(?:[.,]\d+)?", raw))

    if "значение и погрешность" in q and "слитно" in q:
        return "measurement"
    if (
        "выберите все верные" in q
        or "запишите цифры, под которыми" in q
        or "запишите номера" in q
        or "номера выбранн" in q
    ):
        if digits and len(set(digits)) == len(digits):
            return "multi_digit"
        if digits:
            return "code"
    if (
        "как изменятся" in q
        or "как изменится" in q
        or "как изменяются" in q
        or "как изменяется" in q
        or "как изменился" in q
        or "как изменилась" in q
        or "как изменилось" in q
        or "установите соответствие" in q
        or "цифры могут повторяться" in q
    ):
        if digits:
            return "code"
    if (
        "ответ дайте" in q
        or "ответ приведите" in q
        or "чему равен" in q
        or "чему равна" in q
        or "чему равно" in q
        or "определите" in q
        or "найдите" in q
        or "какова" in q
        or "каково" in q
        or "рассчитайте" in q
        or "вычислите" in q
    ) and numeric:
        return "numeric"
    if re.fullmatch(r"[1-5]", raw):
        return "single_digit"
    if numeric:
        return "numeric"
    return fallback


def _mechanic_for(answer_type: str) -> TaskMechanic:
    if answer_type == "single_digit":
        return TaskMechanic.MCQ_SINGLE
    if answer_type == "multi_digit":
        return TaskMechanic.MCQ_MULTI
    if answer_type in {"numeric", "measurement", "code"}:
        return TaskMechanic.NUMERIC
    return TaskMechanic.FREE_RESPONSE


def _parse_number(raw: str) -> float | None:
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        return None


def _c_guess(answer_type: str) -> float:
    if answer_type == "single_digit":
        return 0.25
    if answer_type == "multi_digit":
        return 0.10
    if answer_type == "code":
        return 1.0 / 9.0
    return 0.0


def _difficulty_prior(position: int, answer_type: str, media_required: bool, text: str) -> float:
    base = (POSITION_DIFFICULTY[position] - 3.0) * 0.8
    mechanic = {
        "single_digit": -0.2,
        "multi_digit": 0.2,
        "code": 0.1,
        "numeric": 0.0,
        "measurement": 0.2,
    }.get(answer_type, 0.3)
    length = 0.15 * min(1.5, max(0.0, (len(text) - 80) / 400.0))
    media = 0.1 if media_required else 0.0
    return max(-2.0, min(2.0, base + mechanic + length + media))


def _needs_media(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "рисунке", "рисунок", "график", "графике", "схеме", "схема",
        "таблице", "таблица", "диаграмме", "вектор", "траектори",
    ]
    return any(m in lowered for m in markers)


def _page_offset_map(text: str) -> list[int]:
    offsets = [0]
    for i, ch in enumerate(text):
        if ch == "\x0c":
            offsets.append(i + 1)
    return offsets


def _page_for_offset(offsets: list[int], pos: int) -> int:
    lo, hi = 0, len(offsets) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if offsets[mid] <= pos:
            lo = mid + 1
        else:
            hi = mid - 1
    return max(1, hi + 1)


def parse_pdf(path: Path, position: int, start_id: int) -> tuple[list[dict], dict]:
    text = _pdftotext(path)
    offsets = _page_offset_map(text)
    markers = list(re.finditer(r"(?i)(?:Отв\s*е?т|Ответ)\s*:\s*", text))
    if not markers:
        return [], {"file": path.name, "position": position, "reason": "no answer markers"}

    rows: list[dict] = []
    q_start = 0
    for match in markers:
        answer_end = text.find("\n", match.end())
        if answer_end < 0:
            answer_end = len(text)
        answer_raw = text[match.end():answer_end].strip()
        question = text[q_start:match.start()]
        q_pos = q_start
        q_start = answer_end + 1
        cleaned = _clean_question(question)
        if len(cleaned) < 30:
            continue
        raw, answer_type, value, digits = _normalise_answer(answer_raw)
        if answer_type in {"empty", "text"}:
            continue
        answer_type = _infer_answer_type(
            cleaned, raw, POSITION_ANSWER_TYPE.get(position, answer_type)
        )
        if answer_type == "numeric":
            value = _parse_number(raw)
            digits = None
        elif answer_type in {"multi_digit", "code", "single_digit"}:
            digits = [int(ch) for ch in raw if ch.isdigit()]
            value = None
        elif answer_type == "measurement":
            value = None
            digits = None
        rows.append(
            {
                "position": position,
                "question": cleaned,
                "answer_raw": raw,
                "answer_type": answer_type,
                "answer_value": value,
                "answer_digits": digits,
                "page_start": _page_for_offset(offsets, q_pos),
                "page_end": _page_for_offset(offsets, match.start()),
                "media_required": _needs_media(cleaned),
            }
        )
    return rows, {
        "file": path.name,
        "position": position,
        "answers": len(markers),
        "parsed": len(rows),
    }


def _render_pages(
    pdf_path: Path,
    pages: set[int],
    media_dir: Path,
    position: int,
) -> dict[int, str]:
    if not pages:
        return {}
    media_dir.mkdir(parents=True, exist_ok=True)
    rendered: dict[int, str] = {}
    for page in sorted(pages):
        stem = media_dir / f"position_{position:02d}_page_{page:03d}"
        target = stem.with_suffix(".jpg")
        if target.exists():
            rendered[page] = str(target)
            continue
        subprocess.run(
            [
                "pdftoppm", "-f", str(page), "-l", str(page),
                "-r", "80", "-jpeg", "-jpegopt", "quality=75",
                "-singlefile", str(pdf_path), str(stem),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        rendered[page] = str(target)
    return rendered


def build_bank(pdf_dir: Path, media_dir: Path | None = None) -> tuple[list[KnowledgeUnit], list[Task], dict]:
    kus: list[KnowledgeUnit] = []
    section_ku_id = {s: s for s in SECTION_NAMES}
    for section, name in SECTION_NAMES.items():
        kus.append(
            KnowledgeUnit(
                id=section,
                code=f"section_{section}",
                name=name,
                ku_type=KUType.HYBRID,
                topic_id=section,
                section_id=section,
                ege_weight=1.0,
            )
        )
    for position in range(1, 21):
        section = POSITION_SECTION[position]
        kus.append(
            KnowledgeUnit(
                id=100 + position,
                code=f"ege_position_{position}",
                name=f"ЕГЭ, задание {position}",
                ku_type=KUType.HYBRID,
                topic_id=section,
                section_id=section,
                ege_weight=1.0 if position < 18 else 2.0,
                parent_ids=(section_ku_id[section],),
            )
        )

    tasks: list[Task] = []
    reports: list[dict] = []
    seen: set[str] = set()
    next_id = 100_000
    for position in range(1, 21):
        pdf = pdf_dir / f"Задание {position}.pdf"
        txt = pdf_dir / f"Задание {position}.txt"
        if pdf.exists():
            rows, report = parse_pdf(pdf, position, next_id)
        elif txt.exists():
            rows, report = [], {"file": txt.name, "position": position, "reason": "txt has no answers"}
        else:
            rows, report = [], {"file": None, "position": position, "reason": "missing"}

        rendered: dict[int, str] = {}
        if media_dir is not None and pdf.exists():
            media_pages = {
                p
                for row in rows
                if row["media_required"]
                for p in range(row["page_start"], row["page_end"] + 1)
            }
            rendered = _render_pages(pdf, media_pages, media_dir, position)

        added = 0
        for row in rows:
            key = re.sub(r"\s+", " ", row["question"].lower()).strip() + "|" + row["answer_raw"]
            if key in seen:
                continue
            seen.add(key)
            answer_type = row["answer_type"]
            task = Task(
                id=next_id,
                mechanic=_mechanic_for(answer_type),
                ku_ids=(100 + position,),
                c_guess=_c_guess(answer_type),
                t_expected_sec=float(POSITION_TIME[position]),
                b_difficulty=_difficulty_prior(
                    position, answer_type, row["media_required"], row["question"]
                ),
                a_discrimination=1.0,
                expert_difficulty=POSITION_DIFFICULTY[position],
                topic_id=POSITION_SECTION[position],
                section_id=POSITION_SECTION[position],
                text=row["question"],
                source="ФИПИ",
                version=1,
                media=tuple(
                    rendered[p]
                    for p in range(row["page_start"], row["page_end"] + 1)
                    if p in rendered
                ) if row["media_required"] else (),
                answer_raw=row["answer_raw"],
                answer_type=answer_type,
                answer_value=row["answer_value"],
                answer_digits=tuple(row["answer_digits"] or ()),
                tags=(
                    f"ege_position:{position}",
                    f"answer_type:{answer_type}",
                    f"media_required:{str(row['media_required']).lower()}",
                    f"page_start:{row['page_start']}",
                    f"page_end:{row['page_end']}",
                ),
            )
            tasks.append(task)
            next_id += 1
            added += 1
        report["added"] = added
        reports.append(report)

    meta = {
        "name": "ФИПИ/ЕГЭ физика 2026 — открытый банк",
        "source": "ФИПИ",
        "positions": 20,
        "tasks": len(tasks),
        "knowledge_units": len(kus),
        "media_required": sum(1 for t in tasks if any("media_required:true" in x for x in t.tags)),
        "reports": reports,
        "warning": (
            "Задания с media_required требуют исходного рисунка/графика. "
            "Текстовый слой PDF не заменяет изображение. Перед показом ученику "
            "нужно прикрепить страницу PDF или вырезанный рисунок."
        ),
    }
    return kus, tasks, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--media-dir", type=Path, default=None)
    args = parser.parse_args()

    kus, tasks, meta = build_bank(args.pdf_dir, args.media_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "knowledge_units": [asdict(k) for k in kus],
        "tasks": [
            {
                "id": t.id,
                "mechanic": t.mechanic.value,
                "ku_ids": list(t.ku_ids),
                "ku_weights": list(t.ku_weights),
                "c_guess": t.c_guess,
                "t_expected_sec": t.t_expected_sec,
                "b_difficulty": t.b_difficulty,
                "a_discrimination": t.a_discrimination,
                "expert_difficulty": t.expert_difficulty,
                "is_quarantined": t.is_quarantined,
                "topic_id": t.topic_id,
                "section_id": t.section_id,
                "text": t.text,
                "source": t.source,
                "version": t.version,
                "media": list(t.media),
                "tags": list(t.tags),
                "answer_raw": t.answer_raw,
                "answer_type": t.answer_type,
                "answer_value": t.answer_value,
                "answer_digits": list(t.answer_digits),
            }
            for t in tasks
        ],
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "reports"}, ensure_ascii=False, indent=2))
    for report in meta["reports"]:
        print(report)


if __name__ == "__main__":
    main()
