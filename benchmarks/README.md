# Qualitative Red-Team Benchmark

`qualitative_redteam_cases.yaml` defines safe, abstract red-team cases for the
public demo. The deterministic generator creates sample artifacts under
`reports/demo_benchmark/` without calling external APIs.

Generate demo artifacts:

```bash
python -m src.generate_demo_benchmark
```
