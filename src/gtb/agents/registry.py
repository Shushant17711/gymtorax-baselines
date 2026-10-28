"""Method name -> agent class (imported lazily so torch/sb3 load only when needed)."""
from __future__ import annotations

import importlib

AGENTS = {
    "dummy": "gtb.agents.dummy:DummyAgent",
    "ppo": "gtb.agents.sb3:PPOAgent",
    "sac": "gtb.agents.sb3:SACAgent",
    "mbrl": "gtb.agents.mbrl:MBRLAgent",
    "pimbrl": "gtb.agents.mbrl:PIMBRLAgent",
}


def get_agent_cls(name: str):
    if name not in AGENTS:
        raise KeyError(f"unknown agent {name!r}; known: {sorted(AGENTS)}")
    mod, cls = AGENTS[name].split(":")
    return getattr(importlib.import_module(mod), cls)
