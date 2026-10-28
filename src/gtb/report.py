"""Collect results/ into tables, hypothesis verdicts and figures (used by scripts/make_figures.py)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gtb.eval.metrics import iqm_diff_ci, stratified_bootstrap_ci

LEARNED = ("ppo", "sac", "mbrl", "pimbrl")
REFERENCE = ("random", "scripted", "classical")
LABEL = {"random": "Random", "scripted": "Scripted", "classical": "Classical (tuned)", "ppo": "PPO", "sac": "SAC",
         "mbrl": "MBRL", "pimbrl": "PI-MBRL"}
SPLITS = ("nominal", "heldout", "ood")
COMPONENTS = ("fusion_gain", "h98", "q_min", "q_edge")


def load_results(root: str | Path = "results") -> dict:
    """{method: {"runs": [{"seed", "eval", "curve"}]}}; references have one run (random: seeds pooled)."""
    root = Path(root)
    out: dict = {}
    for m in LEARNED:
        runs = []
        for ev in sorted((root / m).glob("*/seed*/eval.json")):
            curve = []
            mf = ev.parent / "metrics.jsonl"
            if mf.exists():
                curve = [json.loads(x) for x in mf.read_text().splitlines()]
                curve = [x for x in curve if "eval" in x]
            runs.append({"seed": int(ev.parent.name[4:]), "hash": ev.parent.parent.name,
                         "eval": json.loads(ev.read_text()), "curve": curve})
        if runs:
            hashes = {r["hash"] for r in runs}
            if len(hashes) > 1:
                raise ValueError(f"{m}: several config hashes {hashes}; move stale runs out of results/")
            out[m] = {"runs": runs}
    for m in REFERENCE:
        f = root / "reference" / m / "eval.json"
        if f.exists():
            out[m] = {"runs": [{"seed": 0, "eval": json.loads(f.read_text()), "curve": []}]}
    return out


def return_matrix(method: dict, split: str, key: str = "return") -> np.ndarray:
    """(n_runs, n_scenarios); random's pooled seeds are reshaped to (seeds, scenarios)."""
    rows = []
    for r in method["runs"]:
        eps = r["eval"]["episodes"][split]
        vals = np.array([e[key] for e in eps], float)
        rows.append(vals)
    m = np.stack(rows)
    if m.shape[0] == 1 and len(method["runs"][0]["eval"]["episodes"][split]) > 0:
        n_sc = len({json.dumps(e["variant"], sort_keys=True) for e in method["runs"][0]["eval"]["episodes"][split]})
        if n_sc < m.shape[1]:
            m = m.reshape(-1, n_sc)
    return m


def summary_table(res: dict, n_boot: int = 2000) -> dict:
    tab: dict = {}
    for m, d in res.items():
        tab[m] = {}
        for sp in SPLITS:
            R = return_matrix(d, sp)
            p, lo, hi = stratified_bootstrap_ci(R, n_boot=n_boot)
            ep = [e for r in d["runs"] for e in r["eval"]["episodes"][sp]]
            tab[m][sp] = {"iqm": p, "lo": lo, "hi": hi, "n_runs": R.shape[0],
                          "components": {c: float(np.mean([e[f"ret_{c}"] for e in ep])) for c in COMPONENTS},
                          "frac_q_min_lt_1": float(np.mean([e["frac_q_min_lt_1"] for e in ep])),
                          "frac_q95_lt_3": float(np.mean([e["frac_q95_lt_3"] for e in ep])),
                          "fail_rate": float(np.mean([e["failed"] for e in ep]))}
    return tab


def _curve_returns(method: dict, steps: int, split: str) -> np.ndarray | None:
    rows = []
    for r in method["runs"]:
        pts = [x for x in r["curve"] if x["steps"] >= steps and "returns" in x and split in x["returns"]]
        if not pts:
            return None
        rows.append(np.asarray(min(pts, key=lambda x: x["steps"])["returns"][split], float))
    return np.stack(rows)


def verdicts(res: dict, n_boot: int = 2000, first_eval: int = 10000) -> dict:
    def test(x, y):
        if x is None or y is None:
            return {"verdict": "missing"}
        d, lo, hi = iqm_diff_ci(x, y, n_boot=n_boot)
        v = "supported" if lo > 0 else "contradicted" if hi < 0 else "inconclusive"
        return {"diff": d, "lo": lo, "hi": hi, "verdict": v}

    if "pimbrl" not in res or "mbrl" not in res:
        return {h: {"verdict": "missing"} for h in ("H1", "H2", "H3")}
    pi, mb = res["pimbrl"], res["mbrl"]
    return {
        "H1": {"what": "PI-MBRL > MBRL, OOD, final", **test(return_matrix(pi, "ood"), return_matrix(mb, "ood"))},
        "H2": {"what": f"PI-MBRL > MBRL, held-out, first eval (>= {first_eval} steps)",
               **test(_curve_returns(pi, first_eval, "heldout"), _curve_returns(mb, first_eval, "heldout"))},
        "H3": {"what": "PI-MBRL > MBRL, held-out, final",
               **test(return_matrix(pi, "heldout"), return_matrix(mb, "heldout"))},
    }


def latex_table(tab: dict) -> str:
    lines = [r"\begin{tabular}{l" + "c" * len(SPLITS) + "cc}", r"\toprule",
             "Method & " + " & ".join(s.replace("heldout", "held-out").upper() if s == "ood" else s.replace(
                 "heldout", "held-out").capitalize() for s in SPLITS) + r" & $q_{\min}<1$ & fail \\", r"\midrule"]
    for m in (*REFERENCE, *LEARNED):
        if m not in tab:
            continue
        cells = [f"{tab[m][s]['iqm']:.3f} [{tab[m][s]['lo']:.3f}, {tab[m][s]['hi']:.3f}]" for s in SPLITS]
        lines.append(f"{LABEL[m]} & " + " & ".join(cells) +
                     f" & {tab[m]['heldout']['frac_q_min_lt_1']:.2f} & {tab[m]['heldout']['fail_rate']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def verdict_markdown(v: dict) -> str:
    out = ["# Hypothesis verdicts (PREREGISTRATION.md)", "",
           "| | test | IQM diff [95% CI] | verdict |", "|---|---|---|---|"]
    for h, d in v.items():
        ci = f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]" if "diff" in d else "n/a"
        out.append(f"| {h} | {d.get('what', '')} | {ci} | {d['verdict']} |")
    return "\n".join(out) + "\n"


def make_figures(res: dict, tab: dict, out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    made = []
    methods = [m for m in (*REFERENCE, *LEARNED) if m in tab]

    fig, axes = plt.subplots(1, len(SPLITS), figsize=(4 * len(SPLITS), 3.2), sharey=True)
    for ax, sp in zip(axes, SPLITS, strict=True):
        y = np.arange(len(methods))
        p = np.array([tab[m][sp]["iqm"] for m in methods])
        err = np.array([[p[i] - tab[m][sp]["lo"], tab[m][sp]["hi"] - p[i]] for i, m in enumerate(methods)]).T
        ax.barh(y, p, xerr=err, color=["0.6" if m in REFERENCE else "C0" for m in methods])
        ax.set_yticks(y, [LABEL[m] for m in methods])
        ax.set_title(sp.replace("heldout", "held-out"))
        ax.set_xlabel("IQM return")
    fig.tight_layout()
    made.append(out / "iqm_bars.pdf")
    fig.savefig(made[-1])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    for m in LEARNED:
        if m not in res:
            continue
        steps = sorted({x["steps"] for r in res[m]["runs"] for x in r["curve"]})
        if not steps:
            continue
        mean = [np.mean([x["eval"]["heldout"]["iqm"] for r in res[m]["runs"] for x in r["curve"]
                         if x["steps"] == s and "heldout" in x["eval"]]) for s in steps]
        ax.plot(steps, mean, marker="o", ms=3, label=LABEL[m])
    for m in ("scripted", "classical"):
        if m in tab:
            ax.axhline(tab[m]["heldout"]["iqm"], ls="--", c="0.5" if m == "scripted" else "0.2", label=LABEL[m])
    ax.set_xscale("log")
    ax.set_xlabel("env steps")
    ax.set_ylabel("held-out IQM return")
    ax.legend(fontsize=7)
    fig.tight_layout()
    made.append(out / "learning_curves.pdf")
    fig.savefig(made[-1])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(len(methods))
    bottom = np.zeros(len(methods))
    for c in COMPONENTS:
        v = np.array([tab[m]["heldout"]["components"][c] for m in methods])
        ax.bar(x, v, bottom=bottom, label=c)
        bottom += v
    ax.set_xticks(x, [LABEL[m] for m in methods], rotation=30, ha="right")
    ax.set_ylabel("mean return (held-out)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    made.append(out / "components.pdf")
    fig.savefig(made[-1])
    plt.close(fig)
    return made
