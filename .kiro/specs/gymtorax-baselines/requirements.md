# Requirements: Baselines for RL plasma control in Gym-TORAX

## Goal
Publish the first broad, reproducible baseline study on Gym-TORAX (arXiv 2510.11283,
`gymtorax==1.1.1`, `torax==1.4.3`): classical, model-free and model-based controllers on
`gymtorax/IterHybrid-v0`, plus one physics-informed model-based improvement. The comparison
itself is the contribution; the improvement is optional upside.

## Facts measured on this machine (2026-10-28)
- IterHybrid-v0: 150 control steps/episode, dict action (ECRH 3, Ip 1, NBI 3 = 7 dims),
  dict obs (~100 profiles x 25-27 pts + ~80 scalars). Reward = weighted fusion gain, H98, q_min, q_edge.
- The env is **deterministic**: bundled `IterHybridAgent` scores 3.263 on seeds 0 and 1;
  random policy 3.18. The return band is narrow, so components and constraints must be reported.
- Cost: ~0.04 s/step on CPU after a ~5 s JIT; 20 cores, 23 GB RAM, RTX 4060 (jax is CPU-only; torch may use GPU).

## Requirements
1. **Environment interface**
   1.1 A wrapper flattens the obs to a fixed, documented feature vector and maps a [-1,1]^7 action to the env's Box bounds.
   1.2 Reward components (fusion gain, H98, q_min, q_edge) and constraint flags (q_min<1, q95<3) are logged every step.
   1.3 Perturbed variants (initial density/temperature scale, Ip target, transport multipliers) exist so that
       generalisation, not memorisation of one deterministic trajectory, is measured.
2. **Baselines**
   2.1 Reference: random, bundled IterHybridAgent (scripted), tuned PID.
   2.2 Model-free: PPO and SAC.
   2.3 Model-based: learned ensemble dynamics + CEM-MPC (PETS-style).
   2.4 Improvement: a physics-informed dynamics model (0-D power-balance prior + learned residual) used by the same planner.
3. **Protocol**
   3.1 Every method trains under one env-step budget and is evaluated on nominal and perturbed variants.
   3.2 >=5 training seeds per learned method; report IQM with 95% stratified-bootstrap CIs.
   3.3 Budgets, metrics, variants and success thresholds are frozen in `PREREGISTRATION.md` before main runs.
4. **Reproducibility**
   4.1 One config file per run; config hash in every results path; `uv run pytest` passes on CPU in < 3 min.
   4.2 `scripts/make_figures.py` regenerates every table and figure from `results/`.
5. **Dissemination**
   5.1 Public GitHub repo (MIT), pushed after every milestone; paper draft in `paper/`.
