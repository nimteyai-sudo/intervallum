# API reference

Run:

```bash
uvicorn intervallum_ml.api:app --reload
```

Interactive docs: `http://127.0.0.1:8000/docs`.

The reference API is a single-process, in-memory adapter.  Production must
replace `InMemoryStore` with PostgreSQL/Redis and add authentication.

## Health

`GET /health`

```json
{"status": "ok", "knowledge_units": 12, "tasks": 60}
```

## Tasks

`GET /v1/tasks`

Returns the task bank with ML parameters.

## Select next task

`POST /v1/select`

```json
{
  "user_id": "u1",
  "session_position": 0,
  "session_length": 10,
  "consecutive_errors": 0,
  "session_elapsed_sec": 0,
  "review_count": 0,
  "days_to_exam": 40,
  "teacher_assignment_ids": [],
  "recent_task_ids": [],
  "recent_mechanics": [],
  "recent_ku_sets": [],
  "device_width": 390,
  "seed": 42
}
```

Response:

```json
{
  "task": {"id": 50, "mechanic": "match_order", "ku_ids": [1, 2]},
  "score": {
    "task_id": 50,
    "utility": 0.74,
    "zpd": 0.61,
    "memory": 0.0,
    "exam": 0.43,
    "diversity_penalty": 0.0,
    "exploration": 0.02,
    "p_correct": 0.78,
    "p_star": 0.71,
    "theta_task": 0.2,
    "reason": "exam_value"
  },
  "propensity": 0.12,
  "candidates": 60,
  "ranked": []
}
```

`propensity` is the probability that the softmax policy would choose this task.
Persist it with the attempt for IPS/DR evaluation.

## Grade an answer

`POST /v1/grade`

```json
{"task_id": 100000, "answer": "3"}
```

Response:

```json
{
  "task_id": 100000,
  "raw_score": 1.0,
  "correct": true,
  "normalized_user": "3",
  "normalized_correct": "3",
  "method": "single_digit",
  "needs_manual_review": false
}
```

## Submit attempt

`POST /v1/attempt`

```json
{
  "user_id": "u1",
  "task_id": 50,
  "session_id": "s1",
  "timestamp": "2026-10-31T10:00:00Z",
  "raw_score": 1.0,
  "answer": "3",
  "duration_ms": 42000,
  "self_grade": 3,
  "hints_used": [],
  "hint_weight_sum": 0.0,
  "pos_in_session": 2,
  "session_elapsed_sec": 180,
  "consecutive_errors": 0,
  "propensity_score": 0.12
}
```

`self_grade` uses FSRS values: 1 Again, 2 Hard, 3 Good, 4 Easy.
If `raw_score` is omitted, the server grades `answer` with
`intervallum_ml.grading.grade_answer` and uses the resulting score.

Response:

```json
{
  "y_eff": 1.0,
  "g_sync": 3,
  "p_correct_before": 0.71,
  "p_star_before": 0.71,
  "mastery_updates": [
    {"ku_id": 1, "before_theta": 0.1, "after_theta": 0.18,
     "before_sigma": 0.9, "after_sigma": 0.87, "responsibility": 0.62}
  ],
  "memory_updates": [
    {"ku_id": 1, "rating": 3, "stability_before": 4.2,
     "stability_after": 10.1, "due_at": "2026-11-10T10:00:00+00:00"}
  ],
  "diagnostics": {"theta_task": 0.2, "sigma_task": 0.9, "y_eff": 1.0, "g_sync": 3.0}
}
```

## Readiness

`POST /v1/readiness`

```json
{
  "user_id": "u1",
  "exam_date": "2027-06-01T00:00:00Z",
  "blueprint": [
    {"position": 1, "max_points": 1, "ku_ids": [1, 2], "b_exam": 0.0, "a_exam": 1.0}
  ]
}
```

Response:

```json
{
  "expected_raw": 26.0,
  "raw_low": 18.7,
  "raw_high": 33.3,
  "expected_scaled": 48.1,
  "scaled_low": 34.6,
  "scaled_high": 61.7,
  "confidence": 0.35,
  "variance": 31.2
}
```

## Production contract

The TypeScript fast path should mirror these endpoints and calculations.  The
Python service is for offline calibration, simulation and reference testing; it
should not sit synchronously in the lesson request path.
