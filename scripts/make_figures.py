"""results/ -> results/summary.json, results/verdicts.md, paper/table_main.tex, figures/*.pdf

  uv run python scripts/make_figures.py [--root results] [--out figures] [--trajectories]
"""
import argparse
import json
from pathlib import Path

from gtb.report import latex_table, load_results, make_figures, summary_table, verdict_markdown, verdicts


def trajectories(res: dict, out: Path) -> Path:
    """Nominal-scenario trajectories (q_min, T_e[0], Q_fusion) for scripted, classical and seed-0 learned runs."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from gtb.agents import reference
    from gtb.agents.registry import get_agent_cls
    from gtb.envs.variants import nominal
    from gtb.envs.wrapper import FEATURE_NAMES, load_norm
    from gtb.eval.rollout import run_jobs
    from gtb.report import LABEL, LEARNED

    fs = {"scripted": reference.scripted_policy}
    for m in LEARNED:
        if m in res:
            r0 = min(res[m]["runs"], key=lambda r: r["seed"])
            fs[m] = get_agent_cls(m).policy_factory(str(Path("results") / m / r0["hash"] / f"seed{r0['seed']}"))
    names = list(fs)
    recs = run_jobs([(fs[n], nominal(), 0, "configs/env/norm.json", True) for n in names], workers=len(names))
    norm = load_norm()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, k in zip(axes, ("q_min", "T_e[0]", "Q_fusion"), strict=True):
        i = FEATURE_NAMES.index(k)
        for n, r in zip(names, recs, strict=True):
            ax.plot(np.asarray(r["traj"]["obs"])[:, i] * norm["std"][i] + norm["mean"][i], label=LABEL[n])
        ax.set_title(k)
        ax.set_xlabel("t [s]")
    axes[0].axhline(1.0, c="k", lw=0.5, ls=":")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "trajectories.pdf")
    return out / "trajectories.pdf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--table", default="paper/table_main.tex")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--trajectories", action="store_true", help="also roll out nominal trajectories (needs CPU)")
    a = ap.parse_args()
    res = load_results(a.root)
    if not res:
        raise SystemExit(f"no results under {a.root}")
    tab = summary_table(res, n_boot=a.n_boot)
    v = verdicts(res, n_boot=a.n_boot)
    root = Path(a.root)
    (root / "summary.json").write_text(json.dumps({"table": tab, "verdicts": v}, indent=1))
    (root / "verdicts.md").write_text(verdict_markdown(v))
    Path(a.table).parent.mkdir(parents=True, exist_ok=True)
    Path(a.table).write_text(latex_table(tab))
    made = make_figures(res, tab, Path(a.out))
    if a.trajectories:
        made.append(trajectories(res, Path(a.out)))
    print(verdict_markdown(v))
    print("wrote", [str(p) for p in made])


if __name__ == "__main__":
    main()
