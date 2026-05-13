"""CLI for guarded red-team demo/live evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.runner import run_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run red-team evaluation from a YAML config.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_from_config(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
