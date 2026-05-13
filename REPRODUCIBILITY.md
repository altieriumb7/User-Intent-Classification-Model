# Reproducibility Checklist

- Python: 3.11 or 3.12
- Install: `python -m pip install -r requirements.txt`
- Tests: `python -m pytest tests -q`
- Streamlit: `streamlit run app.py`
- Demo benchmark: `python -m src.generate_demo_benchmark`
- Red-team CLI: `python -m src.run_redteam --config evals/config.yaml`
- Docker: `docker build -t llm-redteam .`
- Docker run: `docker run --rm -p 8501:8501 -e DEMO_MODE=true -e ALLOW_LIVE_RUNS=false llm-redteam`
- Hugging Face SDK: Docker
- Hugging Face port: 8501

Environment variables:

- `DEMO_MODE=true`
- `ALLOW_LIVE_RUNS=false`
- `DEFAULT_CONFIG_PATH=evals/config.yaml`
- `REPORTS_DIR=reports`
- `BENCHMARK_MODE=demo`
- `OPENAI_API_KEY=` optional local secret

Expected generated files:

- `reports/demo_benchmark/demo_results.json`
- `reports/demo_benchmark/demo_summary.csv`
- `reports/demo_benchmark/qualitative_case_gallery.md`

Known limitations:

- Public demo uses static deterministic artifacts.
- Hugging Face free storage is ephemeral.
- Live provider execution is guarded but not implemented in this checkout.
- Docker runtime verification requires a running Docker daemon.
