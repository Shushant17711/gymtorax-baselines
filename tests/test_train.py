import json

import pytest

from gtb.config import load_config
from gtb.train import run_dir_for


def test_run_dir_excludes_seed():
    a = load_config(["configs/base.yaml", "configs/agent/dummy.yaml"], ["seed=1"])
    b = load_config(["configs/base.yaml", "configs/agent/dummy.yaml"], ["seed=2"])
    assert run_dir_for(a).parent == run_dir_for(b).parent and run_dir_for(a).name == "seed1"


@pytest.mark.slow
def test_dummy_smoke_run(tmp_path):
    from gtb.train import train

    cfg = load_config(["configs/base.yaml", "configs/agent/dummy.yaml"],
                      ["train.steps=300", "train.eval_every=200", "train.periodic_splits=[nominal]",
                       "train.eval_workers=4"])
    rd = train(cfg, root=str(tmp_path))
    for f in ("config.yaml", "metrics.jsonl", "model.json", "eval.json"):
        assert (rd / f).exists(), f
    lines = [json.loads(x) for x in (rd / "metrics.jsonl").read_text().splitlines()]
    assert any("episode_return" in x for x in lines)
    assert sum("eval" in x for x in lines) == 2  # one periodic (at 200) + final
    ev = json.loads((rd / "eval.json").read_text())
    assert ev["env_steps"] == 300 and set(ev["summary"]) == {"nominal", "heldout", "ood"}
    assert ev["summary"]["heldout"]["n"] == 15
    assert train(cfg, root=str(tmp_path)) == rd  # second call is a no-op
