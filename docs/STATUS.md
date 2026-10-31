# Implementation status

## Done in this repository

- [x] FSRS-6 memory wrapper with deterministic mode and fallback.
- [x] Telemetry scrubber: `y_eff`, objective `g_sync`, hint/time/guessing penalties.
- [x] Conjunctive 3PL IRT: softmin, `p_star`, `p_correct`, guessing floor.
- [x] Online credit assignment with responsibility weights.
- [x] Procedural mastery state and uncertainty updates.
- [x] Pedagogical selector: ZPD, review urgency, exam value, diversity,
      prerequisite gate, cooldown, teacher assignment, review quota,
      controlled exploration, propensity logging.
- [x] EGE readiness projection with 80% interval and confidence.
- [x] Cold start: meta-difficulty, adaptive placement, new-KU priors.
- [x] Offline calibration of `b`, `a`, `tau`, `gamma` with analytic gradients.
- [x] Bad-item detection: point-biserial, low discrimination, slow interaction.
- [x] Exam blueprint anchoring.
- [x] Leakage-safe user/time split, Brier/log-loss/ECE/AUC, IPS, DR, ESS.
- [x] Synthetic world generator and recovery test.
- [x] Versioned JSON model artefacts and JSONB state serialisation.
- [x] FastAPI reference service with select/attempt/readiness.
- [x] Optional FSRS weight optimizer wrapper.
- [x] 83 tests, including property-based tests and synthetic parameter recovery.
- [x] Real EGE/FIPI content bank: 1,145 tasks, 26 KUs, 471 media pages.
- [x] Deterministic answer grading for all imported answer types.
- [x] Stress test: 2,000 selections / 200 sessions / 2,000 attempts / 200 quota checks, 0 errors.
- [x] Policy benchmark: adaptive selector beats random in 30/30 synthetic users.
- [x] Behavioural student simulator with latent knowledge, hints, fatigue, dropout and edge profiles.
- [x] Real-bank simulation: adaptive +68% true gain vs random; true p 0.73 vs 0.54.
- [x] Dockerfile, compose, Makefile, CI.

## Deliberately deferred

- [ ] TypeScript fast-path port (production online path).
- [ ] PostgreSQL schema and migrations.
- [ ] Redis cache and distributed rate limiting.
- [ ] Authentication and MAX initData validation.
- [ ] Teacher/parent/admin UI.
- [ ] Content CMS and review workflow.
- [ ] Official EGE raw→scaled table (the default is linear).
- [ ] Positions 6 and 18 in the content bank: source PDFs have no reliable answer layer.
- [ ] Per-user FSRS parameter optimization in production (wrapper exists;
      requires `torch`/`pandas`).
- [ ] Off-policy tuning of selector weights (IPS/DR scaffolding exists).
- [ ] Deep knowledge tracing (DKT/SAKT) — not justified before enough data.
- [ ] A/B experiment service and guardrail automation.
- [ ] 152-ФЗ legal review and school data-processing agreements.

## Why some things are deferred

The ML core should be **honest** about what it can support at launch.  A deep
knowledge-tracing model on 60–90 students would overfit and would not be
explainable to teachers.  The implemented hybrid is the correct first system:
FSRS + conjunctive IRT + deterministic pedagogical policy + calibrated
forecast.  Complexity is added only when the logs justify it.
