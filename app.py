"""Streamlit demo for support-ticket intent classification."""

from __future__ import annotations

import json

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
from src.predict_intent import predict_text
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


st.set_page_config(page_title="User Intent Classifier", layout="wide")
model = ensure_artifacts()
data = load_demo_data()
report = load_report()
metrics = load_metrics()
routing_report = load_routing_report()

st.title("User Intent Classification")
st.caption("Demo baseline: TF-IDF + Logistic Regression trained on clearly marked synthetic support-ticket data.")

with st.sidebar:
    st.header("Preloaded examples")
    selected = st.radio("Choose a message", EXAMPLES, label_visibility="collapsed")
    st.divider()
    st.subheader("Routing map")
    routing_map = pd.DataFrame(
        [{"intent": intent, "support_team": team} for intent, team in INTENT_TO_TEAM.items()]
    )
    st.dataframe(routing_map, use_container_width=True, hide_index=True)

message = st.text_area("Support message", value=selected, height=140)

if message.strip():
    result = predict_text(message, top_k=3)
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted intent", str(result["intent"]).replace("_", " ").title())
    col2.metric("Confidence", format_percent(float(result["confidence"])))
    col3.metric("Suggested team", str(result["support_team"]))

    top_predictions = pd.DataFrame(result["top_predictions"])
    top_predictions["probability"] = top_predictions["probability"].map(format_percent)
    top_predictions["intent"] = top_predictions["intent"].str.replace("_", " ").str.title()
    st.subheader("Top-3 intent probabilities")
    st.dataframe(top_predictions, use_container_width=True, hide_index=True)

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
        st.image(str(CONFUSION_MATRIX_PATH), use_container_width=True)
    else:
        st.info("Run `python -m src.train` to generate the confusion matrix artifact.")

with right:
    st.markdown("#### Per-class precision / recall / F1")
    class_rows = report[report["label"].isin(model.labels)].copy()
    display_report = class_rows[["label", "precision", "recall", "f1-score", "support"]].copy()
    display_report["label"] = display_report["label"].str.replace("_", " ").str.title()
    st.dataframe(display_report, use_container_width=True, hide_index=True)

st.subheader("Example messages by intent")
for intent in sorted(data["intent"].unique()):
    examples = data.loc[data["intent"] == intent, "message"].head(3).tolist()
    with st.expander(f"{intent.replace('_', ' ').title()} -> {route_intent(intent)}"):
        for example in examples:
            st.write(f"- {example}")
