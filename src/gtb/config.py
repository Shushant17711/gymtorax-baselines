"""YAML configs -> frozen nested dataclass-like namespaces, with `a.b=c` overrides and stable hashing."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml


class Config:
    """Immutable nested config with attribute access."""

    __slots__ = ("_d",)

    def __init__(self, d: dict[str, Any]):
        object.__setattr__(self, "_d", MappingProxyType({k: Config(v) if isinstance(v, dict) else v
                                                         for k, v in d.items()}))

    def __getattr__(self, k: str) -> Any:
        try:
            return self._d[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        raise AttributeError("Config is frozen")

    def __getitem__(self, k: str) -> Any:
        return self._d[k]

    def __contains__(self, k: str) -> bool:
        return k in self._d

    def get(self, k: str, default: Any = None) -> Any:
        return self._d.get(k, default)

    def to_dict(self) -> dict[str, Any]:
        return {k: v.to_dict() if isinstance(v, Config) else copy.deepcopy(v) for k, v in self._d.items()}

    def __eq__(self, other) -> bool:
        return isinstance(other, Config) and self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        return f"Config({self.to_dict()!r})"


def _set(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
        if not isinstance(d, dict):
            raise KeyError(f"cannot override inside non-dict at {dotted}")
    d[keys[-1]] = value


def _parse(raw: str) -> Any:
    v = yaml.safe_load(raw)
    if isinstance(v, str):  # YAML 1.1 reads "1e-4" as a string
        try:
            return float(v)
        except ValueError:
            pass
    return v


def apply_overrides(d: dict, overrides: list[str]) -> dict:
    d = copy.deepcopy(d)
    for o in overrides:
        key, sep, raw = o.partition("=")
        if not sep:
            raise ValueError(f"override must be key=value, got {o!r}")
        _set(d, key.strip(), _parse(raw))
    return d


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    d = yaml.safe_load(Path(path).read_text()) or {}
    return Config(apply_overrides(d, overrides or []))


def config_hash(cfg: Config | dict, n: int = 10) -> str:
    d = cfg.to_dict() if isinstance(cfg, Config) else cfg
    blob = json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:n]


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=True))
