from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import yaml


def _merge_dicts(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _to_namespace(data):
    if isinstance(data, dict):
        return SimpleNamespace(**{key: _to_namespace(value) for key, value in data.items()})
    if isinstance(data, list):
        return [_to_namespace(item) for item in data]
    return data


@dataclass
class AppConfig:
    raw: dict
    config_dir: Path

    def __getattr__(self, item):
        try:
            return _to_namespace(self.raw[item])
        except KeyError as exc:
            raise AttributeError(item) from exc

    def for_usecase(self, usecase: str):
        if usecase == "default":
            data = deepcopy(self.raw)
            data["usecase"] = {
                "name": "default",
                "system_prompt": "base_system.md",
            }
            return _to_namespace(data)

        usecase_path = self.config_dir / "usecases" / f"{usecase}.yaml"
        if not usecase_path.exists():
            raise ValueError(f"Unknown use case: {usecase}")

        with usecase_path.open("r", encoding="utf-8") as handle:
            override = yaml.safe_load(handle) or {}

        merged = _merge_dicts(self.raw, override)
        return _to_namespace(merged)


def load_app_config(config_dir: str = "configs") -> AppConfig:
    base_path = Path(config_dir) / "base.yaml"
    with base_path.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle) or {}
    return AppConfig(raw=base, config_dir=Path(config_dir))
