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
- Obs: 14 scalars {Q_fusion, H98, q95, q_min, W_thermal_total, P_fusion, P_aux_total, tau_E, n_e_line_avg,
  fgw_n_e_line_avg, li3, T_e/T_i volume avg, dW_thermal_dt} (Ip is not in the observation; it is the previous action)
  plus T_e, T_i, n_e, q, j_total profiles subsampled to 8 radial points, plus the previous action and t/T.
  62 dims (54 physics features + 7 previous action + t/T); physics features clipped to +-10 after normalisation. Normalisation is a fixed per-feature affine fitted once on 20 random+scripted episodes and saved to `configs/env/norm.json`.
  No running normalisation, so all methods see identical inputs.
- Action: tanh-space a in [-1,1]^7 -> affine to Box; the env's own ramp-rate limits still apply.
- Episode = 150 steps, gamma = 0.99. Failure (TORAX error/NaN) -> terminate with the env's reward and a `failed` flag.

## Variants (generalisation)
`variants.py` subclasses IterHybridEnv and edits its TORAX config: initial T scale [0.8,1.2], nbar [0.75,0.95] (fGW),
pedestal T scale [0.85,1.15], Z_eff [1.7,2.3], QLKNN collisionality multiplier [0.8,1.25] (log-uniform; TORAX has no global
chi multiplier). Training draws from the train range each reset; evaluation uses nominal + 15 held-out in-range draws
+ 5 OOD draws (1-2 params in the band out to twice the train half-width), frozen in `configs/env/heldout.yaml`.
Scripted-agent returns span 3.01-3.53 across draws (nominal 3.263), so variants matter.

## Methods
- PID: the bundled PIDAgent on Ip, gains tuned by Optuna (50 trials) on train variants.
- PPO/SAC: stable-baselines3 (torch), SubprocVecEnv with 8 workers, budget 300k env steps (~1-2 h each). SAC is single-env with UTD 1.
- MBRL: an ensemble of 5 MLPs predicting the delta of the normalised obs; CEM horizon 10, 256 samples, 4 iterations; models train on the GPU, planning runs on CPU in the rollout workers. Data: 15k steps (half uniform-random, half scripted + N(0,0.2) noise), then rounds of 16 parallel MPC episodes (CPU
planning in workers, exploration noise 0.1) with a warm-started refit after each round, total budget 50k steps.
  Planning reward is recomputed from predicted scalars with the env's reward function on the decoded features.
- Physics-informed model (improvement): same ensemble and planner, plus a 0-D power-balance prior on stored energy,
  dW_{t+1} = dW_t exp(-1/tau_E) + k (P_aux,cmd - P_aux,t) (loss linearised as W/tau_E; k fitted on the agent's own
  initial data, ~0.24). The ensemble learns the residual for W and all other features. Hypothesis: better sample
  efficiency and better OOD transfer.

## Metrics
Primary: episodic return on held-out variants. Secondary: reward components, constraint-violation rate, failure rate,
nominal return, and env steps to reach the scripted agent's return. Aggregation: IQM over seeds x variants, 95% CI with 2000 bootstraps.

## Compute plan
Total runs: (PPO, SAC, MBRL, PI-MBRL) x 5 seeds = 20, plus PID tuning; at most ~40 CPU-hours, so they run in parallel on 20 cores over 2-3 days.
