import json
import subprocess
import sys

import numpy as np
import pytest

from gtb.report import COMPONENTS, load_results, summary_table, verdicts

N_SC = {"nominal": 1, "heldout": 15, "ood": 5}


def _episodes(rng, shift: dict, seed_noise=0.0):
    # no seed noise by default: on null splits the methods tie exactly, so the verdict test is deterministic
    # (with noise, a 95% CI on a null difference excludes 0 ~5% of the time)
    out = {}
    for sp, n in N_SC.items():
        eps = []
        for i in range(n):
            r = 3.0 + 0.1 * np.sin(i) + shift.get(sp, 0.0) + seed_noise * rng.normal()
            eps.append({"return": r, "steps": 150, "failed": False, "variant": {"i": i}, "seed": 0,
                        "frac_q_min_lt_1": 0.1, "frac_q95_lt_3": 0.0,
                        **{f"ret_{c}": r / 4 for c in COMPONENTS}})
        out[sp] = eps
    return out


def _write_run(root, method, seed, shift, rng):
    rd = root / method / "abc123" / f"seed{seed}"
    rd.mkdir(parents=True)
    eps = _episodes(rng, shift)
    (rd / "eval.json").write_text(json.dumps({"env_steps": 50000, "summary": {}, "episodes": eps}))
    lines = []
    for steps, frac in ((10000, 0.5), (50000, 1.0)):
        ep = _episodes(rng, {k: v * frac for k, v in shift.items()})
        lines.append({"steps": steps, "eval": {sp: {"iqm": float(np.mean([e["return"] for e in v]))}
                                               for sp, v in ep.items()},
                      "returns": {sp: [e["return"] for e in v] for sp, v in ep.items()}})
    (rd / "metrics.jsonl").write_text("\n".join(json.dumps(x) for x in lines))


@pytest.fixture
def fake_root(tmp_path):
    rng = np.random.default_rng(0)
    for s in range(5):
        _write_run(tmp_path, "mbrl", s, {}, rng)
        _write_run(tmp_path, "pimbrl", s, {"ood": 0.3, "heldout": 0.0}, rng)
        _write_run(tmp_path, "ppo", s, {"ood": -0.2}, rng)
    # seed-to-seed spread on the OOD effect only, so H1's CI is not degenerate
    for s in range(5):
        f = tmp_path / "pimbrl" / "abc123" / f"seed{s}" / "eval.json"
        ev = json.loads(f.read_text())
        for e in ev["episodes"]["ood"]:
            e["return"] += 0.03 * (s - 2)
        f.write_text(json.dumps(ev))
    ref = tmp_path / "reference" / "scripted"
    ref.mkdir(parents=True)
    ref.joinpath("eval.json").write_text(json.dumps({"episodes": _episodes(rng, {}, 0.0)}))
    return tmp_path


def test_verdicts_on_planted_effects(fake_root):
    res = load_results(fake_root)
    assert set(res) == {"mbrl", "pimbrl", "ppo", "scripted"}
    tab = summary_table(res, n_boot=300)
    assert tab["pimbrl"]["ood"]["iqm"] > tab["mbrl"]["ood"]["iqm"] + 0.25
    v = verdicts(res, n_boot=300)
    assert v["H1"]["verdict"] == "supported"
    assert v["H3"]["verdict"] == "inconclusive"
    assert v["H2"]["verdict"] == "inconclusive"  # held-out shift is 0 at every eval


def test_stale_hashes_rejected(fake_root):
    (fake_root / "mbrl" / "zzz" / "seed0").mkdir(parents=True)
    (fake_root / "mbrl" / "zzz" / "seed0" / "eval.json").write_text(
        (fake_root / "mbrl" / "abc123" / "seed0" / "eval.json").read_text())
    with pytest.raises(ValueError, match="several config hashes"):
        load_results(fake_root)


def test_make_figures_script(fake_root, tmp_path):
    out, tex = tmp_path / "figs", tmp_path / "t.tex"
    subprocess.run([sys.executable, "scripts/make_figures.py", "--root", str(fake_root), "--out", str(out),
                    "--table", str(tex), "--n-boot", "200"], check=True, capture_output=True)
    assert {p.name for p in out.iterdir()} == {"iqm_bars.pdf", "learning_curves.pdf", "components.pdf"}
    assert "PI-MBRL" in tex.read_text() and r"\toprule" in tex.read_text()
    assert "supported" in (fake_root / "verdicts.md").read_text()
    assert set(json.loads((fake_root / "summary.json").read_text())["verdicts"]) == {"H1", "H2", "H3"}
