from pathlib import Path

# The root directory of the project.
ROOT_DIR = Path(__file__).resolve().parent.parent

# Dataset directories.
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

# Keep the demo dataset for app testing and demonstrations.
DATA_PATH = RAW_DATA_DIR / "demo_messages.csv"

# Use the cleaned UCI dataset for real-data model training.
UCI_DATA_PATH = PROCESSED_DATA_DIR / "uci_sms_spam.csv"

# Model and evaluation artifact locations.
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "spamguard_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
