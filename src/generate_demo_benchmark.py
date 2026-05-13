"""Generate deterministic demo benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.redteam_benchmark import DEFAULT_CASES_PATH, DEFAULT_OUTPUT_DIR, generate_demo_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic demo benchmark reports.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = generate_demo_benchmark(args.cases, args.output)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
