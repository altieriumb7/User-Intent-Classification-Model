import json

import pytest

from src.redteam_benchmark import generate_demo_benchmark, load_benchmark_cases, summarize_cases
from src.runner import run_from_config
from src.runtime_config import load_runtime_settings


def test_demo_mode_disables_live_execution(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("ALLOW_LIVE_RUNS", "true")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = load_runtime_settings()

    assert settings.demo_mode is True
    assert settings.live_run_allowed("session-key") is False


def test_missing_openai_key_does_not_break_config(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = load_runtime_settings()

    assert settings.openai_api_key_present is False


def test_benchmark_cases_load_and_summarize():
    cases = load_benchmark_cases()
    summary = summarize_cases(cases)

    assert len(cases) == 6
    assert summary["total_cases"] == 6
    assert summary["pass_count"] == 5
    assert summary["warning_count"] == 1
    assert summary["fail_count"] == 0


def test_demo_benchmark_generation_is_deterministic(tmp_path):
    first = generate_demo_benchmark(output_dir=tmp_path / "first")
    second = generate_demo_benchmark(output_dir=tmp_path / "second")

    assert first["json"].read_text(encoding="utf-8") == second["json"].read_text(encoding="utf-8")
    assert first["csv"].read_text(encoding="utf-8") == second["csv"].read_text(encoding="utf-8")
    assert first["markdown"].read_text(encoding="utf-8") == second["markdown"].read_text(encoding="utf-8")


def test_reports_are_written_to_configured_reports_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path))
    monkeypatch.setenv("DEMO_MODE", "true")

    result = run_from_config("evals/config.yaml")
    payload = json.loads((tmp_path / "demo_benchmark" / "demo_results.json").read_text())

    assert result["mode"] == "demo"
    assert payload["summary"]["total_cases"] == 6


def test_live_run_without_key_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.setenv("ALLOW_LIVE_RUNS", "true")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config_path = tmp_path / "live_config.yaml"
    config_path.write_text("mode: live\nreports_dir: reports\n", encoding="utf-8")

    with pytest.raises(PermissionError):
        run_from_config(config_path)
