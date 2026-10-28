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

Status: all code is written and tested (tasks 1-8.1, 9.1). The protocol is frozen in `PREREGISTRATION.md`.
Training is in progress (`scripts/gtb status`). Two full sweeps run, 4 methods x 5 seeds each: one on the
original reward and one with Q capped at 10 (PREREGISTRATION.md, amendment 1).
The plan is in `.kiro/specs/gymtorax-baselines/`.

## Usage

```bash
uv run python scripts/train.py configs/agent/ppo.yaml seed=0           # one run -> results/ppo/<hash>/seed0
uv run python scripts/sweep.py configs/sweep/main.yaml --dry-run       # list the preregistered sweep
uv run python scripts/evaluate.py --reference scripted                 # reference controllers
uv run python scripts/make_figures.py                                  # tables, verdicts, figures
```

## Setup

```bash
uv sync
uv run pytest          # fast tests
uv run pytest -m slow  # tests that run TORAX episodes
```

## Running the study (pause and resume at any time)

```bash
scripts/gtb start     # run everything in the background (original reward, then capped reward)
scripts/gtb status    # progress of every run
scripts/gtb pause     # every job saves a checkpoint and stops (takes a few minutes); then sleep or shut down
scripts/gtb resume    # continue exactly where it stopped
scripts/gtb log       # follow the pipeline log
```

How it works:

- **Checkpoints.** Each training run saves its full state to `<run>/ckpt/` every 5 minutes, and again when
  paused. That state is the model, optimiser, replay buffer (SAC), collected data and RNG state (MBRL), the
  step counter and the evaluation schedule. A checkpoint is written to `ckpt.new` and swapped in only once
  complete, so a crash or power loss mid-save leaves the previous checkpoint intact.
- **Safe points.** Jobs only pause between updates: after a PPO rollout and its update, after a SAC episode,
  or after an MBRL collect-and-refit round. So a pause never splits an episode or an update.
- **Nothing is counted twice.** On resume, `metrics.jsonl` is cut back to where the checkpoint was taken,
  and the step budget is exact. MBRL resumes bit-for-bit (tested). PPO and SAC continue from their saved
  state but draw fresh training scenarios after a resume.
- **Unplanned stops** (crash, power cut, `kill`) lose at most the work since the last checkpoint (≤ 5 min per
  run). `scripts/gtb resume` continues from there, and a crashed job is retried automatically.
- **Laptop sleep** freezes the jobs and they continue on wake. Use `scripts/gtb pause` first if you will shut
  down or reboot: the background service does not survive either.

## License

MIT. See [LICENSE](LICENSE).
