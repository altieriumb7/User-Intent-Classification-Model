"""Project configuration constants."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEMO_DATA_PATH = DATA_DIR / "demo_support_tickets.csv"
MODEL_PATH = MODELS_DIR / "intent_classifier.joblib"
METRICS_PATH = REPORTS_DIR / "evaluation_metrics.json"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.csv"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
ROUTING_REPORT_PATH = REPORTS_DIR / "routing_report.json"
MONITORING_REPORT_PATH = REPORTS_DIR / "monitoring_report.json"
CV_REPORT_PATH = REPORTS_DIR / "cross_validation_report.json"

RANDOM_STATE = 42
TEST_SIZE = 0.10
VALIDATION_SIZE = 0.10
DEFAULT_CONFIDENCE_THRESHOLD = 0.55
