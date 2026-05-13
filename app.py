"""Streamlit demo for intent classification and qualitative red-team benchmarking."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    DEMO_DATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    ROUTING_REPORT_PATH,
)
from src.data import load_dataset
from src.model import load_model
from src.predict_intent import predict_with_model
from src.redteam_benchmark import generate_demo_benchmark, load_benchmark_report
from src.runtime_config import load_runtime_settings
from src.routing import INTENT_TO_TEAM, route_intent
from src.train import train_and_evaluate


EXAMPLES = [
    "I cannot access my account because the two factor code never arrives.",
    "Please refund the duplicate charge on my credit card.",
    "Does your product integrate with Salesforce?",
]


@st.cache_resource(show_spinner=False)
def ensure_artifacts():
    if not MODEL_PATH.exists() or not METRICS_PATH.exists() or not CLASSIFICATION_REPORT_PATH.exists():
        train_and_evaluate()
    return load_model(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_demo_data() -> pd.DataFrame:
    return load_dataset(DEMO_DATA_PATH)


@st.cache_data(show_spinner=False)
def load_report() -> pd.DataFrame:
    if not CLASSIFICATION_REPORT_PATH.exists():
        train_and_evaluate()
    return pd.read_csv(CLASSIFICATION_REPORT_PATH)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, float]:
    if not METRICS_PATH.exists():
        train_and_evaluate()
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_routing_report() -> dict[str, object]:
    if not ROUTING_REPORT_PATH.exists():
        train_and_evaluate()
    return json.loads(ROUTING_REPORT_PATH.read_text(encoding="utf-8"))


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


@st.cache_data(show_spinner=False)
def ensure_demo_benchmark(reports_dir: str) -> dict[str, object]:
    output_dir = Path(reports_dir) / "demo_benchmark"
    json_path = output_dir / "demo_results.json"
    if not json_path.exists():
        generate_demo_benchmark(output_dir=output_dir)
    return load_benchmark_report(json_path)


def render_benchmark_tab(settings) -> None:
    if settings.demo_mode:
        st.info(
            "Public demo mode: live model calls are disabled. This demo uses sample benchmark "
            "reports. Clone the repo and set OPENAI_API_KEY to run full evaluations."
        )
    else:
        st.warning("Live mode is configured. Any enabled live run may consume API credits.")

    report_data = ensure_demo_benchmark(str(settings.reports_dir))
    summary = report_data["summary"]
    cases = report_data["cases"]
    categories = Counter(case["category"] for case in cases)

    metric_cols = st.columns(5)
    metric_cols[0].metric("Total cases", summary["total_cases"])
    metric_cols[1].metric("Pass", summary["pass_count"])
    metric_cols[2].metric("Warning", summary["warning_count"])
    metric_cols[3].metric("Fail", summary["fail_count"])
    metric_cols[4].metric("Pass rate", format_percent(float(summary["pass_rate"])))

    st.subheader("Category breakdown")
    st.dataframe(
        pd.DataFrame(
            [{"category": category, "cases": count} for category, count in sorted(categories.items())]
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Qualitative case gallery")
    case_frame = pd.DataFrame(cases)
    st.dataframe(
        case_frame[
            [
                "case_id",
                "category",
                "status",
                "input_prompt",
                "expected_behavior",
                "observed_output",
                "qualitative_assessment",
                "notes",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    st.subheader("Failure and warning examples")
    flagged = [case for case in cases if case["status"] in {"fail", "warning"}]
    if flagged:
        for case in flagged:
            with st.expander(f"{case['case_id']} - {case['status']} - {case['category']}"):
                st.write(case["qualitative_assessment"])
                st.caption(case["notes"])
    else:
        st.write("No failed or warning cases in the current demo report.")

    output_dir = Path(settings.reports_dir) / "demo_benchmark"
    json_bytes = (output_dir / "demo_results.json").read_bytes()
    csv_bytes = (output_dir / "demo_summary.csv").read_bytes()
    md_bytes = (output_dir / "qualitative_case_gallery.md").read_bytes()
    dl_cols = st.columns(3)
    dl_cols[0].download_button("Download JSON report", json_bytes, "demo_results.json")
    dl_cols[1].download_button("Download CSV summary", csv_bytes, "demo_summary.csv")
    dl_cols[2].download_button("Download Markdown gallery", md_bytes, "qualitative_case_gallery.md")


def render_classifier_tab() -> None:
    model = ensure_artifacts()
    data = load_demo_data()
    report = load_report()
    metrics = load_metrics()
    routing_report = load_routing_report()

    st.title("User Intent Classification")
    st.caption(
        "Demo baseline: TF-IDF + Logistic Regression trained on clearly marked synthetic support-ticket data."
    )

    with st.sidebar:
        st.header("Preloaded examples")
        selected = st.radio("Choose a message", EXAMPLES, label_visibility="collapsed")
        st.divider()
        st.subheader("Routing map")
        routing_map = pd.DataFrame(
            [{"intent": intent, "support_team": team} for intent, team in INTENT_TO_TEAM.items()]
        )
        st.dataframe(routing_map, width="stretch", hide_index=True)

    message = st.text_area("Support message", value=selected, height=140)

    if message.strip():
        result = predict_with_model(message, model, top_k=3)
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted intent", str(result["intent"]).replace("_", " ").title())
        col2.metric("Confidence", format_percent(float(result["confidence"])))
        col3.metric("Suggested team", str(result["support_team"]))

        top_predictions = pd.DataFrame(result["top_predictions"])
        top_predictions["probability"] = top_predictions["probability"].map(format_percent)
        top_predictions["intent"] = top_predictions["intent"].str.replace("_", " ").str.title()
        st.subheader("Top-3 intent probabilities")
        st.dataframe(top_predictions, width="stretch", hide_index=True)

    st.subheader("Evaluation")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Accuracy", format_percent(metrics["accuracy"]))
    metric_cols[1].metric("Macro F1", format_percent(metrics["macro_f1"]))
    metric_cols[2].metric("Micro F1", format_percent(metrics["micro_f1"]))
    metric_cols[3].metric("Misrouting rate", format_percent(float(routing_report["misrouting_rate"])))

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Confusion matrix")
        if CONFUSION_MATRIX_PATH.exists():
            st.image(str(CONFUSION_MATRIX_PATH), width="stretch")
        else:
            st.info("Run `python -m src.train` to generate the confusion matrix artifact.")

    with right:
        st.markdown("#### Per-class precision / recall / F1")
        class_rows = report[report["label"].isin(model.labels)].copy()
        display_report = class_rows[["label", "precision", "recall", "f1-score", "support"]].copy()
        display_report["label"] = display_report["label"].str.replace("_", " ").str.title()
        st.dataframe(display_report, width="stretch", hide_index=True)

    st.subheader("Example messages by intent")
    for intent in sorted(data["intent"].unique()):
        examples = data.loc[data["intent"] == intent, "message"].head(3).tolist()
        with st.expander(f"{intent.replace('_', ' ').title()} -> {route_intent(intent)}"):
            for example in examples:
                st.write(f"- {example}")


st.set_page_config(page_title="LLM Red Team Evaluation Dashboard", layout="wide")
settings = load_runtime_settings()

with st.sidebar:
    st.header("Runtime settings")
    st.write(f"Current mode: `{settings.mode_label}`")
    st.write(f"DEMO_MODE: `{settings.demo_mode}`")
    st.write(f"ALLOW_LIVE_RUNS: `{settings.allow_live_runs}`")
    st.write(f"OPENAI_API_KEY present: `{settings.openai_api_key_present}`")
    st.write(f"DEFAULT_CONFIG_PATH: `{settings.default_config_path}`")
    st.write(f"REPORTS_DIR: `{settings.reports_dir}`")
    st.write(f"Selected config: `{settings.default_config_path}`")
    st.write(f"Selected report: `{settings.reports_dir / 'demo_benchmark/demo_results.json'}`")
    if not settings.demo_mode:
        st.session_state["session_api_key"] = st.text_input(
            "Session-only API key",
            type="password",
            help="Stored only in this Streamlit session; never written to disk.",
        )
    if settings.live_run_allowed(st.session_state.get("session_api_key")):
        st.warning("Live evaluation may consume API credits.")
        if st.button("Run live evaluation (may consume API credits)"):
            st.error("Live provider execution is not implemented in this repository checkout.")

benchmark_tab, classifier_tab = st.tabs(
    ["Benchmark & Qualitative Evaluation", "Existing Intent Classifier Demo"]
)
with benchmark_tab:
    render_benchmark_tab(settings)
with classifier_tab:
    render_classifier_tab()
