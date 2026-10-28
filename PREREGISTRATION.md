# Preregistration: Gym-TORAX baselines

Written before any learned method has been trained (2026-10-28). The protocol below is frozen at the commit that
adds this file; any later change goes in a dated amendment at the bottom, and the paper reports deviations.

## Environment and scenarios
- `gymtorax==1.1.1`, `torax==1.4.3`, task `gymtorax/IterHybrid-v0` (150 one-second control steps), through
  `gtb.envs.wrapper.FlatEnv` (62-dim observation, frozen `configs/env/norm.json`; action in [-1,1]^7).
- Scenario perturbations (`gtb.envs.variants`): initial T scale [0.8,1.2], nbar [0.75,0.95] fGW, pedestal T
  scale [0.85,1.15], Z_eff [1.7,2.3], QLKNN collisionality multiplier [0.8,1.25] (log-uniform). Training draws a
  new scenario every reset from this range.
- Evaluation set `configs/env/heldout.yaml` (seed 20261028, never used for training draws): nominal (1),
  held-out in-range (15), OOD (5, one or two parameters beyond the training range, up to twice the half-width).
  Every method is evaluated with a deterministic policy, one episode per scenario.

## Methods and budgets (env steps per run)
| method | budget | config |
|---|---|---|
| random | none | 5 seeds |
| scripted (bundled IterHybridAgent schedule) | none | deterministic |
| classical (feed-forward + PI on q_min), Optuna-tuned | 120 trials x 5 train scenarios = 90k steps | `scripts/tune_classical.py` seed 0 |
| PPO (SB3, 8 envs) | 300k | `configs/agent/ppo.yaml` |
| SAC (SB3, UTD 1) | 300k | `configs/agent/sac.yaml` |
| MBRL: 5-member ensemble + CEM-MPC | 50k | `configs/agent/mbrl.yaml` |
| PI-MBRL: same + power-balance prior on W | 50k | `configs/agent/pimbrl.yaml` |

Learned methods: seeds 0-4 (`configs/sweep/main.yaml`). Hyperparameters are the config defaults; there is no
per-method hyperparameter search beyond the classical controller's tuning, which is part of that method. MBRL and
PI-MBRL share every setting except the prior.

## Metrics
- Primary: episodic return (the env's reward, unchanged) on each split. Aggregate = IQM over all (seed, scenario)
  pairs of a split, with 95% stratified-bootstrap CIs (2000 resamples, scenarios as strata).
- Secondary: the four reward components, the fraction of steps with q_min < 1 and with q95 < 3, the fraction of
  failed (TORAX error / non-finite) episodes, and the held-out return at each 10k-step evaluation (model-based) or
  25k-step evaluation (model-free) for learning curves.

## Hypotheses and decision rules
Differences are IQM(A) - IQM(B) with the CI from `gtb.eval.metrics.iqm_diff_ci` (seeds resampled independently
per method within each scenario).

- **H1 (primary): PI-MBRL beats MBRL out of distribution.** Supported if the 95% CI of
  IQM(PI-MBRL) - IQM(MBRL) on the OOD split lies entirely above 0. Contradicted if it lies entirely below 0.
  Otherwise inconclusive.
- **H2: PI-MBRL is more sample-efficient.** Same test on the held-out split at the first evaluation (10k env steps).
- **H3: PI-MBRL beats MBRL in distribution.** Same test on the held-out split at the end of training.

Descriptive comparisons, with no claim attached: every learned method against the scripted and tuned-classical
controllers on each split, and model-based against model-free at matched env steps (50k: the PPO/SAC learning
curve evaluated at 50k).

## What would make us stop or amend
- A method fails in more than 20% of the episodes on its training scenarios: we report it as-is. We do not
  retune it unless an amendment is written before looking at its evaluation-set returns.
- An implementation bug found after the runs: fix, rerun every affected method on all seeds, report the bug.
