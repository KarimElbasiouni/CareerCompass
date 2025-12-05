import os
import pandas as pd
from pathlib import Path
from src.config import RAW_JSONL_PATH

def load_jsonl_dataset(path: str | os.PathLike = None) -> pd.DataFrame:
    
    jsonl_path = Path(str(path)) if path else RAW_JSONL_PATH

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL dataset not found at: {jsonl_path}")

    df = pd.read_json(jsonl_path, lines=True)

    print(f"[info] loaded JSONL dataset from {jsonl_path}")
    print(f"[info] shape: {df.shape}")
    print(f"[info] columns: {list(df.columns)}")
    return df


if __name__ == "__main__":
    from pathlib import Path
    from src.config import RAW_JSONL_PATH

    df = load_jsonl_dataset(RAW_JSONL_PATH)
    print(df.head(5))
