# Mathematical notes

## 1. FSRS memory layer

FSRS is a three-component memory model:

- `D` — difficulty, 1..10;
- `S` — stability, days until recall probability reaches 90%;
- `R` — retrievability, probability of recall now.

Forgetting curve (FSRS-6):

```text
R(t, S) = (1 + F * t / S)^(-d)
F = 0.9^(-1/d) - 1
```

At `t = S`, `R = 0.9`.

Desired-retention interval:

```text
I(r, S) = S / F * (r^(-1/d) - 1)
```

When `r = 0.9`, `I = S`.

A review with grade `G` updates `S` and `D`.  Qualitatively:

- a successful recall increases `S`;
- the increase is larger when `R` is low (the student almost forgot);
- the increase is smaller when `S` is already large;
- the increase is smaller for difficult items;
- `Again` decreases `S` and increases `D`.

The package `fsrs` implements the full FSRS-6 equations and the per-user
optimizer.  We do not reimplement the weight update here.

## 2. Conjunctive IRT

A task `i` requires units `K_i`.  The effective ability for unit `k` is:

```text
theta_eff_k = theta_k + gamma * ln(R_k)
```

`gamma >= 0` controls how much memory decay reduces procedural performance.
If the memory state has not started, `R` is treated as 1.

The conjunctive ability is a weighted softmin:

```text
theta_task = -1/tau * log( sum_k w_k * exp(-tau * theta_eff_k) )
```

where `sum_k w_k = 1`.  As `tau -> 0`, this approaches the weighted mean; as
`tau -> infinity`, it approaches the minimum.  `tau = 2` is a useful default.

The probability of a correct answer without guessing:

```text
p_star = sigmoid(a_i * (theta_task - b_i))
```

With guessing floor `c_i`:

```text
p_correct = c_i + (1 - c_i) * p_star
```

`c_i` is 0 for numeric/formula items, 0.25 for 4-option single choice, 0.05 for
multi-select combinations, etc.

## 3. Credit assignment

The responsibility of unit `k` is:

```text
q_k = softmax(-tau * theta_eff_k)_k
```

If a task has two units, one at `theta = +2` and one at `theta = -0.5`, then
`q_weak` is close to 1.  An error is attributed mostly to the weak unit.

The mastery update for an attempt with effective score `y_eff`:

```text
delta = y_eff - p_correct
g_i = (1 - c_i) * p_star / p_correct
theta_k' = theta_k + K * sigma_k^2 * a_i * g_i * q_k * delta
```

`K` is the online learning rate.  The precision update is:

```text
info_gain = a_i^2 * g_i^2 * q_k^2 * p_correct * (1 - p_correct)
sigma_k' = max(sigma_min, 1 / sqrt(1 / sigma_k^2 + info_gain))
```

A repeated identical task discounts the theta step by
`repeat_task_theta_discount`; memory can still update because recognition of the
same item is evidence of memory, not of transfer.

## 4. Pedagogical utility

```text
p_target =
    0.85  if session_position <= 1
    0.88  if consecutive_errors >= 2 or elapsed >= 12 min
    0.72  otherwise

ZPD = exp(-(p_star - p_target)^2 / (2 * 0.15^2))

ReviewUrgency = max_k clamp((0.95 - R_k) / 0.25, 0, 1.5)

ExamValue = weighted_mean_k [ V_k * (1 - sigmoid(theta_eff_k)) * urgency_k ]

Utility = 0.35 * ZPD
        + 0.30 * ReviewUrgency
        + 0.25 * ExamValue
        - 0.10 * DiversityPenalty
        + 0.10 * mean(sigma)
        + 0.03 * GumbelNoise
```

`urgency_k` is 1 for a normal timeline; near the exam it rises for mid-mastery
topics and falls for hopeless hard topics.

The review quota is:

```text
review_quota = round(0.40 * session_length)
```

After the quota is exhausted, review-heavy tasks lose their memory bonus.

## 5. Readiness projection

For each EGE position `m` with maximum points `M_m`:

```text
p_m = c_m + (1 - c_m) * sigmoid(a_m * (theta_task_m - b_m))
E[points_m] = M_m * p_m
Var[points_m] = M_m^2 * (p_m(1-p_m) + (a_m p_m(1-p_m) sigma_m)^2)
```

Aggregate:

```text
Raw = sum_m E[points_m]
Var = sum_m Var[points_m]
80% interval = Raw +/- 1.2816 * sqrt(Var)
```

The interval is converted through the raw→scaled table.  The default table is
linear and is a placeholder; production must use the official EGE scale.

## 6. Calibration

The negative log-likelihood for continuous `y_eff`:

```text
L = -[y log p + (1-y) log(1-p)]
```

with priors:

```text
L_total = L + lambda_b * sum_i (b_i - b_meta_i)^2
            + lambda_a * sum_i (a_i - 1)^2
            + lambda_tau * (tau - tau_0)^2
            + lambda_gamma * (gamma - gamma_0)^2
```

Analytic gradients are implemented in `calibration.py`.  With fewer than
`min_attempts_for_tau` / `min_attempts_for_gamma`, the corresponding parameter
is fixed at its prior; the data are not sufficient to identify it.

## 7. Identifiability caveat

`gamma` and `tau` are partially confounded at small sample sizes: both control
the spread of effective abilities.  The implementation therefore fixes them
until enough attempts exist and reports the number of attempts in the model
artefact.  This is a deliberate bias-variance trade-off, not an oversight.
