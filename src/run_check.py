from src.load_dataset import load_jsonl_dataset
from src.config import RAW_JSONL_PATH

def main():
    print("[info] checking setup...")
    print(f"[info] expecting JSONL at: {RAW_JSONL_PATH}")

    try:
        df = load_jsonl_dataset()
        print("[success] dataset loaded.")
        print(df.head(3))
    except FileNotFoundError as e:
        print("[fail]", e)
        print("---> Put your file at that exact path or update src/config.py <---")

if __name__ == "__main__":
    main()
