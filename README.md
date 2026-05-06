# User Intent Classification Model

Portfolio-quality baseline for classifying support messages into user intent categories and routing them to the right support team.

## What It Does

This project classifies support tickets into eight intent categories:

- `billing`
- `technical_issue`
- `refund`
- `account_access`
- `product_question`
- `complaint`
- `cancellation`
- `escalation`

The included model is a reproducible multi-class baseline using TF-IDF features and Logistic Regression. The project also includes evaluation artifacts, cross-validation and monitoring reports, routing simulation, confidence-threshold triage fallback, CLI inference, tests, and a Streamlit demo.

## Data

The repository ships with `data/demo_support_tickets.csv`, a synthetic demo dataset created for this project. It is clearly marked with `is_synthetic=true`.

No real public support-ticket dataset is bundled here. To use a real dataset later, provide a CSV with at least these columns:

- `message`
- `intent`

Then train with:

```bash
python -m src.train --data path/to/your_dataset.csv
```

## Project Structure

```text
data/       Demo support-ticket dataset
src/        Training, evaluation, prediction, preprocessing, and routing code
models/     Saved model artifacts generated after training
reports/    Evaluation and routing reports generated after training
tests/      Pytest suite
app.py      Streamlit demo
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train And Evaluate

```bash
python -m src.train
```

This generates:

- `models/intent_classifier.joblib`
- `reports/evaluation_metrics.json`
- `reports/classification_report.csv`
- `reports/confusion_matrix.png`
- `reports/routing_report.json`
- `reports/cross_validation_report.json`
- `reports/monitoring_report.json`

The evaluation includes accuracy, macro F1, micro F1, per-class precision, per-class recall, per-class F1, and a confusion matrix. The routing simulation maps predicted intents to support teams and computes misrouting rate against ground truth labels. Training now also writes a cross-validation report and a monitoring snapshot (average confidence, low-confidence rate, intent distribution).

## Current Demo Results

The checked-in reports were generated from the included synthetic demo dataset:

| Metric | Value |
| --- | ---: |
| Accuracy | 1.00 |
| Macro F1 | 1.00 |
| Micro F1 | 1.00 |
| Misrouting rate | 0.00 |

These results are only a sanity check for the demo pipeline. They should not be interpreted as real-world performance because the evaluation set contains synthetic examples.

## CLI Inference

```bash
python -m src.predict_intent --text "I cannot access my account" --confidence-threshold 0.55
```

Example output:

```json
{
  "intent": "account_access",
  "confidence": 0.42,
  "top_predictions": [
    {"intent": "account_access", "probability": 0.42},
    {"intent": "technical_issue", "probability": 0.18},
    {"intent": "escalation", "probability": 0.11}
  ],
  "support_team": "Identity & Access Support"
}
```

Confidence values depend on the trained model and dataset split.

## Streamlit Demo

```bash
streamlit run app.py
```

The demo shows:

- message input box
- predicted intent
- confidence score
- top-3 intent probabilities
- suggested support team routing
- confusion matrix
- per-class precision / recall / F1
- example messages for each intent category
- preloaded example messages

If model artifacts are missing, the app trains the baseline once from the included demo data.

## Tests

```bash
pytest
```

The tests cover preprocessing, prediction, routing logic, and evaluation metrics.

## Routing Map

| Intent | Support Team |
| --- | --- |
| `account_access` | Identity & Access Support |
| `billing` | Billing Operations |
| `cancellation` | Retention & Account Changes |
| `complaint` | Customer Experience |
| `escalation` | Escalation Desk |
| `product_question` | Product Specialists |
| `refund` | Refunds & Credits |
| `technical_issue` | Technical Support |

## Limitations

- The bundled dataset is synthetic demo data, not a real production corpus.
- Reported metrics are computed only on the included demo split unless you train on a real dataset.
- The baseline is intended to be easy to inspect and reproduce, not production-ready.
- No claim is made about business impact such as reduced misrouting unless computed from a real deployment dataset.
- A transformer classifier is not included by default to keep the inference path lightweight and reproducible.
