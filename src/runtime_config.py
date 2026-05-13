"""Runtime settings for safe dashboard and CLI execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RuntimeSettings:
    demo_mode: bool
    allow_live_runs: bool
    openai_api_key_present: bool
    default_config_path: Path
    reports_dir: Path
    benchmark_mode: str

    @property
    def mode_label(self) -> str:
        return "demo" if self.demo_mode else "live"

    def live_run_allowed(self, session_api_key: str | None = None) -> bool:
        return (
            not self.demo_mode
            and self.allow_live_runs
            and (self.openai_api_key_present or bool(session_api_key))
        )


def load_runtime_settings() -> RuntimeSettings:
    return RuntimeSettings(
        demo_mode=env_bool("DEMO_MODE", True),
        allow_live_runs=env_bool("ALLOW_LIVE_RUNS", False),
        openai_api_key_present=bool(os.getenv("OPENAI_API_KEY")),
        default_config_path=Path(os.getenv("DEFAULT_CONFIG_PATH", "evals/config.yaml")),
        reports_dir=Path(os.getenv("REPORTS_DIR", "reports")),
        benchmark_mode=os.getenv("BENCHMARK_MODE", "demo"),
    )
