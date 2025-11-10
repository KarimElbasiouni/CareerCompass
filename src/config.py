import os
from pathlib import Path

# project root = directory above src
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# data folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# make sure these exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# canonical data + feature paths
RAW_JSON_PATH = RAW_DATA_DIR / "resumes_dataset.jsonl"  # line-delimited JSON
PROCESSED_PARQUET = PROCESSED_DATA_DIR / "resumes_v1.parquet"
SPLITS_JSON = PROCESSED_DATA_DIR / "splits_v1.json"
FEATURES_DIR = DATA_DIR / "features"

# feature defaults
DEFAULT_FEATURE_MODE = "tfidf"  # or "transformer"
TRANSFORMER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # light, contextual
MIN_TITLE_FREQ = 40  # collapse rarer titles into "Other" later
SEED = 42

# backwards compatibility with existing helpers
RAW_JSONL_PATH = RAW_JSON_PATH
RANDOM_SEED = SEED
os.makedirs(FEATURES_DIR, exist_ok=True)
