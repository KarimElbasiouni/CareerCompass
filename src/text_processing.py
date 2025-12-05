import re
import pandas as pd

TEXT_COLUMNS = ["Summary", "Experience", "Education", "Skills", "Text"]

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"(\+?\d[\d\s\-\(\)]{6,}\d)")

def scrub_pii(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    return text

def scrub_columns(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    if cols is None:
        cols = TEXT_COLUMNS
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(scrub_pii)
    return df

def add_text_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    combined = []

    for _, row in df.iterrows():
        parts = []
        for col in TEXT_COLUMNS:
            val = row.get(col)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
        joined = "\n".join(parts)
        joined = scrub_pii(joined)
        combined.append(joined)

    df["text_clean"] = combined
    print("[text_processing] added text_clean")
    return df

if __name__ == "__main__":
    pass
