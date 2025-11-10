# src/label_creation.py

import pandas as pd

# map raw categories to a cleaner label
CATEGORY_NORMALIZATION = {
    "software engineer": "Software Engineer",
    "java developer": "Java Developer",
    "data scientist": "Data Scientist",
    "accountant": "Accountant",
    "project manager": "Project Manager",
    # add more as you see them
}

# map cleaner label to a family
FAMILY_LOOKUP = {
    "Software Engineer": "Computers / IT",
    "Java Developer": "Computers / IT",
    "Data Scientist": "Computers / IT",
    "Accountant": "Business / Finance",
    "Project Manager": "Management",
}

def normalize_category(raw_cat: str) -> str:
    if not isinstance(raw_cat, str):
        return "Other"
    t = raw_cat.strip().lower()
    return CATEGORY_NORMALIZATION.get(t, raw_cat.strip())  # fallback to original nicely stripped

def to_family(clean_label: str) -> str:
    return FAMILY_LOOKUP.get(clean_label, "Other")

def add_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Category" in df.columns:
        df["y_title"] = df["Category"].apply(normalize_category)
    else:
        df["y_title"] = "Other"

    df["y_family"] = df["y_title"].apply(to_family)

    print("[label_creation] added y_title and y_family")
    return df

if __name__ == "__main__":
    pass
