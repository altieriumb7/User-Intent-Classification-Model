"""Deterministic qualitative benchmark loading and report generation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CASES_PATH = Path("benchmarks/qualitative_redteam_cases.yaml")
DEFAULT_OUTPUT_DIR = Path("reports/demo_benchmark")


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    category: str
    input_prompt: str
    expected_behavior: str
    observed_output: str
    qualitative_assessment: str
    status: str
    notes: str


def load_benchmark_cases(path: Path | str = DEFAULT_CASES_PATH) -> list[BenchmarkCase]:
    case_path = Path(path)
    payload = yaml.safe_load(case_path.read_text(encoding="utf-8")) or {}
    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError(f"Benchmark file must contain a list under 'cases': {case_path}")

    loaded: list[BenchmarkCase] = []
    for item in cases:
        loaded.append(
            BenchmarkCase(
                case_id=str(item["id"]),
                category=str(item["category"]),
                input_prompt=str(item["input_prompt"]),
                expected_behavior=str(item["expected_behavior"]),
                observed_output=str(item["demo_output"]),
                qualitative_assessment=str(item["qualitative_assessment"]),
                status=str(item["status"]).lower(),
                notes=str(item.get("notes", "")),
            )
        )
    return loaded


def summarize_cases(cases: list[BenchmarkCase]) -> dict[str, Any]:
    total = len(cases)
    pass_count = sum(case.status == "pass" for case in cases)
    fail_count = sum(case.status == "fail" for case in cases)
    warning_count = sum(case.status == "warning" for case in cases)
    return {
        "total_cases": total,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "warning_count": warning_count,
        "pass_rate": pass_count / total if total else 0.0,
        "categories_covered": sorted({case.category for case in cases}),
    }


def cases_to_rows(cases: list[BenchmarkCase]) -> list[dict[str, str]]:
    return [
        {
            "case_id": case.case_id,
            "category": case.category,
            "input_prompt": case.input_prompt,
            "expected_behavior": case.expected_behavior,
            "observed_output": case.observed_output,
            "qualitative_assessment": case.qualitative_assessment,
            "status": case.status,
            "notes": case.notes,
        }
        for case in cases
    ]


def render_markdown_report(cases: list[BenchmarkCase], summary: dict[str, Any]) -> str:
    lines = [
        "# Qualitative Red-Team Demo Benchmark",
        "",
        "These are deterministic demo results generated from repository benchmark cases. No external APIs were called.",
        "",
        f"- Total cases: {summary['total_cases']}",
        f"- Pass: {summary['pass_count']}",
        f"- Warning: {summary['warning_count']}",
        f"- Fail: {summary['fail_count']}",
        f"- Pass rate: {summary['pass_rate']:.2%}",
        "",
        "| Case | Category | Status | Assessment |",
        "|---|---|---|---|",
    ]
    for case in cases:
        lines.append(
            f"| {case.case_id} | {case.category} | {case.status} | {case.qualitative_assessment} |"
        )
    lines.append("")
    for case in cases:
        lines.extend(
            [
                f"## {case.case_id}: {case.category}",
                "",
                f"Input prompt: {case.input_prompt}",
                "",
                f"Expected behavior: {case.expected_behavior}",
                "",
                f"Observed/demo output: {case.observed_output}",
                "",
                f"Notes: {case.notes}",
                "",
            ]
        )
    return "\n".join(lines)


def generate_demo_benchmark(
    cases_path: Path | str = DEFAULT_CASES_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    cases = load_benchmark_cases(cases_path)
    summary = summarize_cases(cases)
    rows = cases_to_rows(cases)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "demo_results.json"
    csv_path = out_dir / "demo_summary.csv"
    md_path = out_dir / "qualitative_case_gallery.md"

    json_path.write_text(
        json.dumps({"mode": "demo", "summary": summary, "cases": rows}, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(render_markdown_report(cases, summary), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": md_path}


def load_benchmark_report(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
