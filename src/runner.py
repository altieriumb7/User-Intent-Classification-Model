"""Guarded red-team runner entrypoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from src.live_redteam import DEFAULT_LIVE_MODEL, run_live_benchmark
from src.redteam_benchmark import generate_demo_benchmark
from src.runtime_config import RuntimeSettings, load_runtime_settings


def load_eval_config(path: Path | str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Evaluation config not found: {config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def run_from_config(
    config_path: Path | str,
    settings: RuntimeSettings | None = None,
    session_api_key: str | None = None,
    live_model: str | None = None,
    live_max_cases: int | None = None,
) -> dict[str, Any]:
    settings = settings or load_runtime_settings()
    config = load_eval_config(config_path)
    mode = str(config.get("mode", settings.benchmark_mode)).lower()
    reports_dir = Path(os.getenv("REPORTS_DIR", str(config.get("reports_dir", settings.reports_dir))))

    if mode == "demo" or settings.demo_mode:
        paths = generate_demo_benchmark(
            config.get("benchmark_cases", "benchmarks/qualitative_redteam_cases.yaml"),
            reports_dir / "demo_benchmark",
        )
        return {"mode": "demo", "paths": {key: str(value) for key, value in paths.items()}}

    if not settings.live_run_allowed(session_api_key):
        raise PermissionError(
            "Live runs require DEMO_MODE=false, ALLOW_LIVE_RUNS=true, and OPENAI_API_KEY "
            "or a session-only API key."
        )

    return run_live_benchmark(
        api_key=session_api_key or os.environ["OPENAI_API_KEY"],
        cases_path=config.get("benchmark_cases", "benchmarks/qualitative_redteam_cases.yaml"),
        output_root=reports_dir / "live_benchmark",
        model=str(live_model or config.get("openai_model", os.getenv("OPENAI_MODEL", DEFAULT_LIVE_MODEL))),
        max_cases=int(live_max_cases or config.get("live_max_cases", os.getenv("LIVE_MAX_CASES", "3"))),
    )
