from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError('Config file must contain a JSON object at top level.')
    return data


def apply_config_to_args(parser: argparse.ArgumentParser, argv: list[str] | None = None):
    """Two-stage argparse.

    First read --config if provided, then use config values as parser defaults,
    then parse the full command line so explicit CLI flags still win.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=str, default=None)
    known, _ = pre.parse_known_args(argv)

    if known.config is not None:
        cfg = load_json_config(known.config)
        parser.set_defaults(**cfg)

    parser.add_argument('--config', type=str, default=known.config, help='Path to a JSON config file.')
    return parser.parse_args(argv)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
