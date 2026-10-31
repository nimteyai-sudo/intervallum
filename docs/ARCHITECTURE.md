# Architecture

## Design goals

1. **Learning per minute**, not engagement or immediate correctness.
2. **Explainable** decisions for students, teachers and parents.
3. **Small-data first**: works with zero attempts and improves with logs.
4. **Fast path independent of Python**: the online selector must not make a network call to an ML service.
5. **No leakage**: every logged attempt stores the pre-update state and the action propensity.

## Component flow

```
Attempt arrives
  │
  ├─ TelemetryScrubber
  │    y_eff, g_sync
  │
  ├─ MasteryEngine.predict
  │    theta_task, p_star, p_correct, responsibilities
  │
  ├─ MasteryEngine.update
  │    theta, sigma per KU
  │
  ├─ MemoryScheduler.review
  │    FSRS state for declarative/hybrid KUs
  │
  └─ AttemptResult
       diagnostics for logs and UI

Next task request
  │
  ├─ CandidateFilter
  │    quarantine, cooldown, prerequisite gate, teacher assignment
  │
  ├─ PedagogicalSelector
  │    ZPD + memory + exam + diversity + exploration
  │
  └─ SelectionResult
       chosen task, score components, propensity
```

## Layer contracts

### Memory

- Input: `MemoryState`, `Rating`, `now`.
- Output: new `MemoryState` with `stability`, `difficulty`, `due_at`.
- FSRS-6 from the MIT `fsrs` package.
- Fallback fixed intervals are available if the dependency is absent.

### Skill

- Input: task, per-KU states.
- Output: `TaskPrediction` with `theta_task`, `p_star`, `p_correct`, `responsibilities`.
- The softmin implements conjunctive requirements.
- `theta_eff = theta + gamma * ln(R)` connects memory decay to procedural performance.

### Policy

- Input: user states, session context, task bank.
- Output: ranked `TaskScore`s and a chosen task with a propensity.
- Hard filters are separate from utility.  A teacher assignment bypasses the
  prerequisite gate but still gets a scaffold recommendation.
- The review quota prevents “review hell” after a break.

### Forecast

- Input: states, 26-position blueprint, exam date, scale table.
- Output: expected raw score, 80% interval, scaled score interval, confidence.
- The projection uses effective theta **at the exam date**, so forgetting is
  included.

## Data flow and storage

| Store | Contents | Why |
|---|---|---|
| PostgreSQL | users, tasks, attempts, review logs, assignments, consents | transactional source of truth |
| Redis | hot user state, task bank cache, rate limits | low-latency fast path |
| Object storage | Parquet attempts, model artefacts | offline training and audit |
| ClickHouse (later) | event analytics | large-scale dashboards |

The ML tables use `ml_user_uuid`, not MAX IDs.  The mapping table is isolated.

## Calibration and deployment

```
attempts ──► ItemCalibrator ──► ModelArtifact (JSON)
                                   │
                             shadow mode
                                   │
                         guardrails + A/B
                                   │
                             active version
```

Nightly guardrails:

- max absolute change in item difficulty < 0.5;
- Spearman correlation between old and new difficulties > 0.92;
- cohort mean readiness change < 1.5 points;
- calibration ECE < 0.035.

If a guardrail fails, keep the old artefact and alert.

## Failure modes

### Review hell

A returning student sees a large overdue backlog.  The selector caps review
share and prioritises high-EGE-value units.

### Template memorisation

A student recognises a task by its numbers.  Repeated identical tasks discount
the mastery update; numeric templates should regenerate parameters.

### Mechanic exploitation

Easy mechanics inflate mastery.  Expert priors and IRT difficulty calibration
keep mechanic difficulty separated.

### Prerequisite deadlock

A teacher assigns a topic whose prerequisite is weak.  Teacher assignments are
never hard-blocked; the system inserts a short scaffold instead.

### Scale drift

Online updates can drift the theta/b scale.  Item parameters update only in the
nightly batch, with an anchored mean and guardrails.

## Roadmap

### Phase 1 — launch (0–50k attempts)

- knowledge graph and item metadata;
- FSRS-6, telemetry scrubber, IRT update;
- pedagogical selector with fixed weights;
- readiness projection with uncertainty;
- logging `state_before` and `propensity_score`.

### Phase 2 — calibration (50k–300k)

- nightly `b_i`, `a_i`, `tau` calibration;
- bad-item detection;
- exam-scale anchoring on mock exams;
- within-student block experiment.

### Phase 3 — scale (300k+)

- per-cohort FSRS parameters;
- `Q` matrix refinement;
- off-policy weight tuning;
- deep knowledge tracing only if justified.
