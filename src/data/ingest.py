"""Ingest the raw resumes JSONL into a normalized, de-identified parquet."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.config import PROCESSED_PARQUET, RAW_JSON_PATH
from src.labels.normalize import map_title_to_family, normalize_title
from src.utils.pii import mask_email_phone, mask_exact_name

SECTION_FIELDS = ("Summary", "Experience", "Skills", "Education")
TARGET_COLUMNS = ("resume_id", "text_norm", "title_raw", "family_raw", "source")


def _to_text(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk.strip()
    return ""


def _collapse_whitespace(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.split())


def _build_text_norm(row: Dict[str, Any]) -> str:
    parts: list[str] = []
    for field in SECTION_FIELDS:
        value = _to_text(row.get(field))
        if value:
            parts.append(value)
    if not parts:
        fallback = _to_text(row.get("Text"))
        if fallback:
            parts.append(fallback)
    combined = "\n\n".join(parts)
    combined = mask_email_phone(combined)
    combined = mask_exact_name(combined, _to_text(row.get("Name")) or None)
    return _collapse_whitespace(combined)


def _extract_resume_id(row: Dict[str, Any], idx: int) -> str:
    raw_id = row.get("ResumeID")
    if isinstance(raw_id, str) and raw_id.strip():
        return raw_id.strip()
    if raw_id is not None and not pd.isna(raw_id):
        return str(raw_id)
    return f"row_{idx}"


def ingest(input_path: Path, output_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    raw_df = pd.read_json(input_path, lines=True, dtype=False)
    processed: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_df.to_dict(orient="records")):
        text_norm = _build_text_norm(row)
        if not text_norm:
            continue
        title = normalize_title(_to_text(row.get("Category")))
        processed.append(
            {
                "resume_id": _extract_resume_id(row, idx),
                "text_norm": text_norm,
                "title_raw": title,
                "family_raw": map_title_to_family(title),
                "source": _to_text(row.get("Source")) or None,
            }
        )

    processed_df = pd.DataFrame(processed, columns=TARGET_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_parquet(output_path, index=False)
    return processed_df


def _print_preview(df: pd.DataFrame) -> None:
    preview = df.head(3).to_dict(orient="records")
    print("[ingest] sample rows:")
    for row in preview:
        snippet = row["text_norm"][:120]
        print(
            f"  - id={row['resume_id']} title={row['title_raw']} "
            f"source={row['source']} text='{snippet}...'"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize raw resume JSONL.")
    parser.add_argument(
        "--in",
        dest="input_path",
        default=str(RAW_JSON_PATH),
        help="Path to the raw JSONL file",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        default=str(PROCESSED_PARQUET),
        help="Destination parquet path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    print(f"[ingest] loading raw resumes from {input_path}")
    processed_df = ingest(input_path, output_path)
    print(f"[ingest] saved {len(processed_df)} rows -> {output_path}")
    if not processed_df.empty:
        _print_preview(processed_df)


if __name__ == "__main__":
    main()
