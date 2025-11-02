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

# raw dataset file
RAW_JSONL_PATH = RAW_DATA_DIR / "resumes_dataset.jsonl"

# global seed
RANDOM_SEED = 42
