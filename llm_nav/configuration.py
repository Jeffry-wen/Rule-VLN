"""YAML configuration loading with command-line overrides."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return data


def flatten_sections(config: dict[str, Any], *sections: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for section in sections:
        values = config.get(section, {})
        if values:
            if not isinstance(values, dict):
                raise ValueError(f"Configuration section '{section}' must be a mapping")
            result.update(values)
    return result


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must use KEY=VALUE syntax: {item}")
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return config


def common_config_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to a YAML configuration file.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a YAML value. Nested keys use dot notation.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    return parser
