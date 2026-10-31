# Evaluation protocol

This document is a starting point, not a substitute for a pre-registration.
Before running a confirmatory experiment, freeze the hypothesis, primary
metric, decision rule and sample size in a separate question document.

## Hypotheses

**H1 — adaptive selection.**  The full pedagogical selector increases
30-day retention on held-out knowledge units compared with a topic-order
baseline, at equal or lower study time.

**H2 — conjunctive credit assignment.**  Softmin credit assignment improves
mastery calibration for compound tasks compared with equal attribution across
all required units.

**H3 — readiness projection.**  The projected raw-score interval is calibrated:
among students with a 60–80% projected interval, the observed mock-exam score
falls inside the interval at least 70% of the time.

**H4 — gamification.**  Competitive mechanics improve D7 retention but do not
reduce anchor mastery gain; if they reduce learning, they are disabled by
default for teacher-led classes.

## Primary metric

`Anchor Mastery Gain per Hour`:

```text
gain = (post-test accuracy on unseen anchor items - pre-test accuracy)
       / total study hours
```

Anchor items are sampled from a reserved pool and are not used by the adaptive
policy.  This avoids selection bias: the policy cannot choose the test.

## Design

### Phase 1 — pilot

- 60–90 students, 2–3 classes.
- 4–6 weeks.
- Within-student block randomization across two equivalent topic pools.
- Every student experiences both the full policy and the baseline on different
  topics; the order is randomized.
- Weekly 3-item anchor checks.

### Phase 2 — confirmatory

- Student-level stratified randomization.
- Stratify by grade, baseline diagnostic, and teacher.
- Primary outcome: anchor mastery gain per hour.
- Secondary: session completion, 30-day retention, dropout, satisfaction.
- Guardrails: review workload, hint abuse, session abandonment.

## Splits

- **Offline:** 80% users train, 20% users hold out; within each holdout user,
  first 70% of history for state warm-up, last 30% for forward prediction.
- **Never** split by attempt row.
- **Never** tune on the test horizon.

## Offline metrics and gates

| Metric | Gate |
|---|---|
| Brier score vs item-mean baseline | at least 1.5% improvement |
| ECE | < 0.035 |
| Delayed recall AUC (>5 days) | > 0.72 |
| IPS effective sample size | > 200 |
| Calibration slope | 0.9–1.1 |

A model that improves AUC but worsens calibration is not promoted; the product
uses probabilities for ZPD and readiness intervals.

## Online guardrails

- mean daily review load must not increase by more than 20%;
- session abandonment must not increase by more than 3 percentage points;
- hint abuse rate < 15%;
- no cohort-level readiness shift > 1.5 points after a model update;
- no significant differential item functioning by device width or baseline
  ability.

## Decision rule

Promote the policy only if:

1. the primary metric improves by at least the pre-registered minimum effect;
2. the 95% interval excludes zero;
3. no guardrail is violated;
4. the calibration gate passes.

Otherwise keep the current policy and report the result as null, not as a
motivated subgroup finding.
