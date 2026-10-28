# Design

## Layout
```
src/gtb/            config.py, seed.py
  envs/             wrapper.py (FlatEnv), variants.py (perturbed configs), vec.py
  agents/           reference.py (random, scripted, PID), sb3.py (PPO/SAC), mbrl.py (ensemble+CEM), physics_model.py
  eval/             rollout.py, metrics.py (IQM, bootstrap)
configs/            env/, agent/, sweep/ YAML
scripts/            train.py, evaluate.py, tune_pid.py, sweep.py, make_figures.py
results/<method>/<cfghash>/seed<k>/   metrics.jsonl, eval.json, model
```

## Environment wrapper
- Obs: scalars {Q_fusion, H98, q95, q_min, W_thermal_total, P_fusion, n_e_line_avg, fgw_n_e_line_avg, li3, Ip, T_e/T_i volume avg}
  plus T_e, T_i, n_e, q, j_total profiles subsampled to 8 radial points, plus the previous action and t/T.
  About 60 dims. Normalisation is a fixed per-feature affine fitted once on 20 random+scripted episodes and saved to `configs/env/norm.json`.
  No running normalisation, so all methods see identical inputs.
- Action: tanh-space a in [-1,1]^7 -> affine to Box; the env's own ramp-rate limits still apply.
- Episode = 150 steps, gamma = 0.99. Failure (TORAX error/NaN) -> terminate with the env's reward and a `failed` flag.

## Variants (generalisation)
`variants.py` edits the TORAX config dict before env construction: initial n_e/T_e scale in [0.8,1.2], Ip target +/-10%,
transport chi multiplier in [0.7,1.3]. Training draws from the train range; evaluation uses nominal + 20 fixed held-out draws
(including 5 outside the training range, OOD).

## Methods
- PID: the bundled PIDAgent on Ip, gains tuned by Optuna (50 trials) on train variants.
- PPO/SAC: stable-baselines3 (torch), SubprocVecEnv with 8 workers, budget 300k env steps (~1-2 h each). SAC is single-env with UTD 1.
- MBRL: an ensemble of 5 MLPs predicting the delta of the normalised obs; CEM horizon 10, 256 samples, 4 iterations, in torch on the GPU. Data: 20k random/scripted steps, then iterative MPC data, total budget 50k steps.
  Planning reward is recomputed from predicted scalars with the env's reward function on the decoded features.
- Physics-informed model (improvement): predicts W_thermal via dW/dt = P_heat - W/tau_E with tau_E from the IPB98 scaling,
  then corrects it with a learned residual and models the remaining features by MLP. The hypothesis is better sample efficiency and better OOD transfer.

## Metrics
Primary: episodic return on held-out variants. Secondary: reward components, constraint-violation rate, failure rate,
nominal return, and env steps to reach the scripted agent's return. Aggregation: IQM over seeds x variants, 95% CI with 2000 bootstraps.

## Compute plan
Total runs: (PPO, SAC, MBRL, PI-MBRL) x 5 seeds = 20, plus PID tuning; at most ~40 CPU-hours, so they run in parallel on 20 cores over 2-3 days.
