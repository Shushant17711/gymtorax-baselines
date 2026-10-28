import pytest

from gtb.config import Config, apply_overrides, config_hash, load_config, save_config


def test_roundtrip_override_hash(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("agent:\n  name: ppo\n  lr: 0.0003\nseed: 0\n")
    c = load_config(p, ["agent.lr=1e-4", "seed=3", "env.name=x"])
    assert c.agent.name == "ppo" and c.agent.lr == 1e-4 and c.seed == 3 and c.env.name == "x"
    save_config(c, tmp_path / "o.yaml")
    assert load_config(tmp_path / "o.yaml") == c
    assert config_hash(c) == config_hash(Config(c.to_dict()))
    assert config_hash(c) != config_hash(load_config(p))


def test_hash_key_order_independent():
    assert config_hash({"a": 1, "b": {"c": 2, "d": 3}}) == config_hash({"b": {"d": 3, "c": 2}, "a": 1})


def test_frozen_and_bad_override():
    c = Config({"a": {"b": 1}})
    with pytest.raises(AttributeError):
        c.a = 2
    with pytest.raises(ValueError):
        apply_overrides({}, ["nokey"])


def test_merge_files(tmp_path):
    (tmp_path / "a.yaml").write_text("x: {y: 1, z: 2}\nseed: 0\n")
    (tmp_path / "b.yaml").write_text("x: {z: 5}\n")
    c = load_config([tmp_path / "a.yaml", tmp_path / "b.yaml"], ["seed=2"])
    assert c.x.y == 1 and c.x.z == 5 and c.seed == 2
