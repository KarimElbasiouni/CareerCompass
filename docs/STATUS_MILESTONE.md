<!--
1. Clean dataset script: `build_clean_dataset.py` orchestrates filtering, PII scrub, text concat, and now saves both CSV + Parquet.
2. Canonical parquet path: `data/processed/resumes_v1.parquet` with `resume_id`, `text_norm`, `title_raw`, and legacy `text_clean`, `y_title`, `y_family`.
3. Source text fields feeding `text_norm`: Summary, Experience, Education, Skills, Text (if present per row).
4. `src/text_processing.scrub_columns` and `add_text_clean` handle the regex PII scrubbing before concatenation.
5. Rows missing all important text fields are dropped by `src/data_filter.load_and_filter`.
6. Labels come from `src/label_creation.add_label_columns`, producing `y_title` (→ `title_raw`) and `y_family`.
7. Family labels map via `FAMILY_LOOKUP` with fallback `"Other"` for unmapped titles.
8. Resume IDs roll forward from `ResumeID` where possible, otherwise assigned via `_assign_resume_ids`.
9. `data/processed/resumes_clean.csv` remains as a fallback export for spreadsheet users.
10. TF-IDF builder lives at `src/features/tfidf_build.py` and writes `tfidf_vectorizer.joblib`, `tfidf_X.npz`, `tfidf_index.parquet`.
11. Stratified splits are tracked in `data/processed/splits_v1.json` via `src/utils/splits.ensure_splits`.
12. The LinearSVC baseline/metrics are implemented in `src/models/train_svm.py` with outputs under `models/` + `runs/svm_tfidf/`.
13. Evaluation helpers (`src/eval/metrics.py`) centralize accuracy, macro-F1, confusion plotting, and top-k logic.
14. README §6 describes the canonical commands: clean → TF-IDF → SVM.
15. Advanced transformer work is archived under git tag `vault-IC-data-pipeline-YYYYMMDD` until we revisit it.
-->
