# build_clean_dataset.py

from src.config import PROCESSED_DATA_DIR
from src.data_filter import load_and_filter
from src.text_processing import scrub_columns, add_text_clean
from src.label_creation import add_label_columns

def main():
    df = load_and_filter()
    df = scrub_columns(df)
    df = add_text_clean(df)

    # drop rows with empty text_clean just in case
    df = df[df["text_clean"].str.len() > 0]

    df = add_label_columns(df)

    out_path = PROCESSED_DATA_DIR / "resumes_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"[build_clean_dataset] saved cleaned data to {out_path}")
    print(f"[build_clean_dataset] final shape: {df.shape}")

if __name__ == "__main__":
    main()
