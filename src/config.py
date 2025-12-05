import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"

for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, MODELS_DIR, RUNS_DIR):
    os.makedirs(directory, exist_ok=True)

RAW_JSONL_PATH = RAW_DATA_DIR / "resumes_dataset.jsonl"
PROCESSED_PARQUET = PROCESSED_DATA_DIR / "resumes_v1.parquet"
SPLITS_JSON = PROCESSED_DATA_DIR / "splits_v1.json"

RANDOM_SEED = 42
