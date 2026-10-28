# Implementation Plan - Gym-TORAX baselines (gtb)

Build order: scaffold -> env wrapper/variants -> evaluation harness -> reference controllers ->
model-free -> model-based -> physics-informed improvement -> preregistered sweep -> figures/paper.
"Done" = tests pass under `uv run pytest`, and the task's artefacts are committed and pushed.

## Milestones
- After task 3: wrapped env + eval harness + reference numbers (push).
- After task 5: PPO/SAC working end-to-end (push).
- After task 7: both model-based methods working (push).
- After task 9: full results + paper draft (push).

## Task list

- [x] 1. Scaffold the package and test harness
  - [x] 1.1 Package, config, seeding, license
    - `pyproject.toml` (uv, package `gtb`, MIT, deps gymtorax/torch/stable-baselines3/pyyaml/optuna, dev pytest/ruff), MIT `LICENSE`, README with License section, `.gitignore`
    - `src/gtb/config.py`: YAML -> frozen dataclasses, `a.b=c` CLI overrides, `config_hash()`; `src/gtb/seed.py`
    - Tests: config round-trip, override, hash stability
    - _Requirements: 4.1, 5.1_

- [ ] 2. Environment wrapper and variants
  - [ ] 2.1 Implement `FlatEnv`
    - `src/gtb/envs/wrapper.py`: obs feature selection as in design, [-1,1]^7 action map, previous action + t/T features
    - `info` carries reward components, constraint flags, `failed`
    - Tests: obs shape/finite over one scripted episode, action map hits Box corners, reward equals the underlying env's
    - _Requirements: 1.1, 1.2_
  - [ ] 2.2 Fit and freeze normalisation
    - `scripts/fit_norm.py`: 20 random+scripted episodes -> `configs/env/norm.json` (mean/std per feature, std floor)
    - Tests: normalised features of the scripted episode have |z| < 10
    - _Requirements: 1.1_
  - [ ] 2.3 Implement perturbed variants
    - `src/gtb/envs/variants.py`: `make_variant(params)` edits the TORAX config (n/T init scale, Ip target, chi multiplier); `sample_variant(rng, split)` for train/heldout/ood
    - Fixed held-out list `configs/env/heldout.yaml` (15 in-range + 5 OOD)
    - Tests: variant changes the trajectory, nominal variant reproduces 3.263 for the scripted agent, same params mean the same return
    - _Requirements: 1.3_

- [ ] 3. Evaluation harness and reference controllers
  - [ ] 3.1 Rollout and metrics
    - `src/gtb/eval/rollout.py`: run a policy on a list of variants in a process pool, return per-episode records (return, components, violations, failed)
    - `src/gtb/eval/metrics.py`: IQM, stratified bootstrap CI
    - Tests: IQM on known arrays, CI contains the point estimate, rollout of the random policy is reproducible given a seed
    - _Requirements: 3.2_
  - [ ] 3.2 Reference agents and PID tuning
    - `src/gtb/agents/reference.py`: random, scripted (IterHybridAgent), PID wrapper
    - `scripts/tune_pid.py`: Optuna 50 trials on train variants; frozen gains in `configs/agent/pid.yaml`
    - `scripts/evaluate.py` writes `results/reference/*/eval.json`
    - Tests: each agent completes an episode, PID returns actions inside the Box
    - _Requirements: 2.1_

- [ ] 4. Training entrypoint
  - [ ] 4.1 Unified `scripts/train.py`
    - Dispatch on `agent.name`, write `results/<method>/<hash>/seed<k>/` with config, metrics.jsonl, and periodic held-out eval every 25k steps
    - Tests: a 300-step smoke run for a dummy agent produces all files
    - _Requirements: 3.1, 4.1_

- [ ] 5. Model-free baselines
  - [ ] 5.1 PPO and SAC via stable-baselines3
    - `src/gtb/agents/sb3.py`: SubprocVecEnv(8) for PPO, single env for SAC; train variants sampled per reset
    - `configs/agent/{ppo,sac}.yaml` with budget 300k steps
    - Tests: a 2k-step smoke run of each saves a model that `evaluate.py` can load
    - _Requirements: 2.2_

- [ ] 6. Model-based baseline
  - [ ] 6.1 Ensemble dynamics and CEM planner
    - `src/gtb/agents/mbrl.py`: 5-member probabilistic MLP ensemble (delta obs), torch on CUDA if available; CEM H=10, 256x4
    - Reward model: the env reward recomputed from predicted scalars (decoded with norm.json)
    - Tests: the ensemble overfits a tiny dataset; CEM improves the objective on a toy quadratic; recomputed reward matches the env reward on real transitions to within 1e-3
    - _Requirements: 2.3_

- [ ] 7. Physics-informed improvement
  - [ ] 7.1 Power-balance prior model
    - `src/gtb/agents/physics_model.py`: W update with tau_E from IPB98(y,2) plus a learned residual; the other features via MLP; same interface as the ensemble
    - Tests: the prior alone predicts the scripted trajectory's W_thermal with R^2 > 0.5; the model plugs into the CEM planner
    - _Requirements: 2.4_

- [ ] 8. Preregistration and sweep
  - [ ] 8.1 Freeze the protocol
    - `PREREGISTRATION.md`: budgets, seeds, variants, metrics, the hypothesis that PI-MBRL beats MBRL on OOD IQM by a CI-separated margin; record the commit hash
    - `scripts/sweep.py` + `configs/sweep/main.yaml`: queue of (method x seed) with N workers, resumable
    - Tests: sweep dry-run lists 20 jobs; resume skips finished dirs
    - _Requirements: 3.3_
  - [ ] 8.2 Run the main sweep
    - Launch, monitor, and commit `results/**/eval.json` + `results/summary.json`
    - _Requirements: 3.1, 3.2_

- [ ] 9. Figures and paper
  - [ ] 9.1 Figures and tables
    - `scripts/make_figures.py`: learning curves, IQM bars (nominal/held-out/OOD), component breakdown, example trajectories (q, T_e, Q)
    - Tests: runs on a fake results tree
    - _Requirements: 4.2_
  - [ ] 9.2 Paper draft
    - `paper/main.tex` + `refs.bib`: setup, methods, results, limitations (single scenario, deterministic sim, 0-D prior)
    - _Requirements: 5.1_
