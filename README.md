---
title: LLM Red Team Evaluation Dashboard
emoji: 🧪
colorFrom: red
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
---

# LLM Red Team Evaluation Dashboard

Safe public Streamlit demo for qualitative LLM red-team evaluation, with the original support-ticket intent classifier preserved as a secondary demo tab. The Hugging Face Space runs in demo mode by default and uses deterministic sample benchmark artifacts, so it does not require or spend API keys.

## Live Demo on Hugging Face Spaces

Create a Docker Space and push this repository. Use these Space variables:

```text
DEMO_MODE=true
ALLOW_LIVE_RUNS=false
DEFAULT_CONFIG_PATH=evals/config.yaml
REPORTS_DIR=reports
BENCHMARK_MODE=demo
```

Optional secret for private/local live work only:

```text
OPENAI_API_KEY=<your key>
```

The public Space should keep `DEMO_MODE=true` and `ALLOW_LIVE_RUNS=false`.

## Safe Public Demo

In demo mode:

- live provider/judge calls are disabled
- `OPENAI_API_KEY` is not required
- sample benchmark reports are loaded from `reports/demo_benchmark/`
- generated runtime files use `REPORTS_DIR`; Hugging Face free storage is ephemeral

The dashboard shows whether an API key is present but never displays the key. A session-only key field appears only outside demo mode and is never persisted.

## Local Setup

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m src.generate_demo_benchmark
streamlit run app.py
```

Run tests:

```bash
python -m pytest tests -q
```

## Docker Setup

```bash
docker build -t llm-redteam .
docker run --rm -p 8501:8501 --env-file .env llm-redteam
```

or:

```bash
docker compose up --build
```

Do not commit `.env` or real API keys.

## Benchmark

Cases live in `benchmarks/qualitative_redteam_cases.yaml`. The deterministic demo generator writes:

- `reports/demo_benchmark/demo_results.json`
- `reports/demo_benchmark/demo_summary.csv`
- `reports/demo_benchmark/qualitative_case_gallery.md`

Generate:

```bash
python -m src.generate_demo_benchmark
```

CLI entrypoint:

```bash
python -m src.run_redteam --config evals/config.yaml
```

Current benchmark metrics are demo/sample metrics derived from the repository cases. They are illustrative and should not be reported as live model performance.

## Live Evaluation

This checkout includes a guarded CLI and runtime configuration, but no implemented live provider pipeline. Live execution intentionally fails closed unless:

- `DEMO_MODE=false`
- `ALLOW_LIVE_RUNS=true`
- `OPENAI_API_KEY` or a session-only key is present

Integrate a provider/judge implementation in `src/runner.py` before claiming live results.

## Existing Intent Classifier

The original classifier remains available in the second dashboard tab.

Train/evaluate:

```bash
python -m src.train
```

CLI inference:

```bash
python -m src.predict_intent --text "I cannot access my account" --confidence-threshold 0.55
```

## Security

- No public API keys.
- Demo mode is on by default.
- Live runs are blocked by default.
- User-provided keys are session-only.
- Do not load untrusted `.joblib` files; joblib deserialization can execute code.

## Limitations

- Hugging Face free storage is ephemeral.
- Demo benchmark results are deterministic examples, not live model measurements.
- Full live evaluation requires credentials and a provider/judge implementation.
- The bundled classifier dataset is synthetic demo data.

## Portfolio Summary

This repository demonstrates a deployable Streamlit dashboard, safe-by-default public demo controls, deterministic qualitative benchmark artifacts, Docker/Hugging Face Space packaging, CI, and reproducibility documentation.
