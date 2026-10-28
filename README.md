# gymtorax-baselines

Baselines for reinforcement-learning control of a tokamak plasma in
[Gym-TORAX](https://arxiv.org/abs/2510.11283), which wraps Google DeepMind's
[TORAX](https://github.com/google-deepmind/torax) transport simulator. The task is the ITER hybrid
scenario (`gymtorax/IterHybrid-v0`).

The study compares:

- **Reference controllers:** random, the bundled scripted ITER-hybrid schedule, and a tuned PID.
- **Model-free RL:** PPO and SAC.
- **Model-based RL:** a learned ensemble dynamics model with CEM-MPC.
- **Improvement:** a physics-informed (0-D power-balance) dynamics model inside the same planner.

All methods are evaluated on the nominal scenario, on held-out perturbed scenarios, and on
out-of-distribution scenarios. The nominal environment is deterministic, so a policy evaluated
only there could just memorise one action sequence.

Status: work in progress. The plan is in `.kiro/specs/gymtorax-baselines/` (`requirements.md`,
`design.md`, `tasks.md`).

## Setup

```bash
uv sync
uv run pytest          # fast tests
uv run pytest -m slow  # tests that run TORAX episodes
```

## License

MIT. See [LICENSE](LICENSE).
