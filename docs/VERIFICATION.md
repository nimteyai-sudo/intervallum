# Verification report — 31 October 2026

This report records what was actually run, what failed, what was fixed, and
what remains unverified.  “No defects found” is not the same as “no defects
exist”; the goal here is to make the residual risk explicit.

## Commands and results

```text
pytest -q
83 tests passed

pytest --cov=intervallum_ml
90% statement coverage

mypy --ignore-missing-imports --no-strict-optional src/intervallum_ml scripts
Success: no issues found in 27 source files

ruff check --select F,E9,I src tests scripts
All checks passed

python scripts/stress_test.py --selections 2000 --sessions 200 \
  --attempts 2000 --quota-checks 200 --calibrate
0 errors; 2,000 selections, 200 sessions, 2,000 attempts, 200 quota checks
calibration: Brier 0.190, ECE 0.022

python scripts/policy_benchmark.py --users 30 --kus 20 --tasks 120 --steps 60
random gain 0.0621; adaptive gain 0.1225; adaptive wins 30/30; +97.3%

python scripts/simulate_users.py --students 12 --sessions 4 --tasks 10 \
  --policies random adaptive easy hard topic_order
adaptive true gain 0.0161 vs random 0.0096; adaptive true p 0.730 vs 0.544

python scripts/import_fipi.py ...
1,145 tasks; 26 knowledge units; 471 media tasks; 0 validation errors
8 warnings: section KUs and missing positions 6/18 are unused
```

## Tests that found real defects

1. **Review quota was soft.**  The selector could choose a fifth review-heavy
   task in a ten-task session because ZPD/exam value still outweighed the
   memory penalty.  Fixed with a hard non-review filter after the quota.
   Regression tests cover both the “alternative exists” and “no alternative”
   cases.
2. **Readiness crashed for a blueprint item with no KUs.**  It referenced
   `c_guess` instead of `c_exam`.  Fixed; regression test added.
3. **Answers were lost during import.**  The `Task` contract had no answer
   fields, so the first content bank could not grade anything.  Added
   `answer_raw`, `answer_type`, `answer_value`, `answer_digits`.
4. **Digit answers were misclassified.**  A numeric answer “25” was treated as
   a two-option multi-select; repeated EGE change codes “11/22/333” were
   treated as invalid multi-select sets.  Added a `code` type and
   question-based inference; the official EGE-2026 format is used as a prior.
5. **Measurement answers were dropped.**  Task 19 answers such as `450,012,5`
   were rejected as text.  Added `measurement` type and digit-normalised
   grading.
6. **Task 19 was parsed as only 8/42 tasks.**  Fixed by the measurement type;
   now 42/42.
7. **Task difficulty prior was degenerate.**  All tasks in a position had the
   same `b`, so a new user saw the same success probability for every task.
   Added mechanic/length/media priors; new-user mean `p_correct` is now 0.71.
8. **Reason labels used raw components, not weighted ones.**  A task could be
   labelled “exam value” while ZPD actually dominated the utility.  Fixed.
9. **Duplicate content IDs were silently overwritten.**  Loader now raises on
   duplicate task/KU IDs.
10. **Repeated-task confidence grew without discount.**  The mastery
    uncertainty shrank even when the same task was repeated.  It now uses the
    same discount as the theta update.
11. **Bad-item detection ignored calibrated discrimination.**  It now accepts
    a calibrated `a` map.
12. **API defaulted to synthetic tasks in production.**  It now loads the real
    content bank when present (project root, cwd or `/app/content`).
13. **Mypy found a readiness crash and a model-artefact key mismatch.**  Both
    fixed; mypy is clean.
14. **Grading was missing entirely.**  Added deterministic grading for all
    imported answer types and `POST /v1/grade`.

## Content verification

- 1,145 tasks imported from the downloaded PDFs.
- 26 knowledge units: 6 section nodes + 20 EGE-position nodes.
- 471 tasks with a page render for a figure/graph (334 unique page images).
- 0 structural validation errors.
- All 1,145 tasks grade their own official answer as `1.0`.
- Constructed wrong answers for every type are rejected with `0.0`.
- Positions 6 and 18 are absent: the provided task-6 PDF has no usable text
  layer and the provided task-18 page has no answers.  They are **not**
  fabricated.

## Simulation findings

The behavioural simulator surfaced one product-level issue that unit tests
cannot: the engine's internal `theta` can drift downward even when the true
latent ability improves, because fast correct answers are deliberately given
less credit by the telemetry scrubber.  This is not a scheduling bug, but it
means the student-facing “progress” screen must not show raw `theta` change as
the only progress metric.  Show mastery probability, retention, coverage and
exam readiness instead; keep `theta` internal.

The simulator also showed that a very easy policy can produce learning gains
close to adaptive if the assumed ZPD curve is too flat.  The simulator uses a
sharper ZPD (σ = 0.12) so that the policy comparison is meaningful.

## What is still not proven

- **No real users, no real logs.**  Calibration and policy comparisons use
  synthetic responses.  The numbers demonstrate logic, not educational
  efficacy.
- **Position-level knowledge units.**  The content is linked to one KU per EGE
  position, not to 250 fine-grained topic/skill nodes.  This is a deliberate
  first-pass taxonomy; finer mapping is a content task.
- **Heuristic difficulty priors.**  `b_difficulty` is expert/heuristic until
  real attempts are available.  The calibration pipeline is ready, but it has
  not seen real students.
- **Official EGE scale.**  The readiness converter uses a linear placeholder;
  production must replace it with the official raw→scaled table.
- **Media is page-level.**  Figures are rendered as full PDF pages, not cropped
  task assets.  This is usable for a pilot but not final UI.
- **Production infrastructure.**  The API is in-memory; auth, PostgreSQL,
  Redis, rate limits, audit and teacher/parent cabinets are separate
  workstreams.
- **Source.**  Задания взяты из открытого банка ФИПИ.

## Honest conclusion

The ML core and the recommendation logic are **engineered and heavily tested**,
and the real content bank is loaded and gradeable.  A claim of “100% correct in
production” is not supportable without real users, real attempts, and an
external audit.  What is supportable today: the known defects found by this
verification were fixed, the invariants hold under stress, all imported content
is structurally valid, and the policy beats random order under a known synthetic
learning rule.
