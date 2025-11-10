# CareerCompass

CareerCompass is a resume/job-title analysis and classification project. The goal is to ingest a JSONL dataset of resumes, preprocess text, and build baseline and advanced models that map raw resume text to normalized job titles (and potentially occupational families).

## 1. Repository Layout

```
CareerCompass/
	README.md                <- This file
	requirements.txt         <- Python dependencies
	data/
		raw/                   <- Place original dataset files here (untracked if large)
		processed/             <- Generated cleaned / feature-ready data
	src/
		config.py              <- Paths & constants (PROJECT_ROOT, RAW_JSONL_PATH)
		load_dataset.py        <- Loads JSONL into a pandas DataFrame
		run_check.py           <- Quick environment & dataset existence check
		setup_env.py           <- Prints versions of key libs
	docs/                    <- LaTeX proposal / progress report docs
```

## 2. Prerequisites

- Python 3.10+ (recommended)
- A virtual environment (optional but strongly recommended)
- JSONL resume dataset at path defined by `RAW_JSONL_PATH` in `src/config.py`


Install dependencies (choose one):

**Using venv (standard Python):**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

**Using Conda:**
```bash
conda create -n careercompass python=3.11
conda activate careercompass
pip install -r requirements.txt
```

Verify environment:

```bash
python src/setup_env.py
```

## 3. Placing the Data

1. Put your raw JSONL file at: `data/raw/resumes_dataset.jsonl` (default `RAW_JSONL_PATH`).
2. Each line must be a valid JSON object, e.g.:
	 ```json
	 {"id": "REAL_0001", "text": "10 yrs software design...", "title": "Software Engineer"}
	 {"id": "REAL_0002", "text": "Enthusiastic Java developer...", "title": "Java Developer"}
	 ```
3. If you use a different filename, update `RAW_JSONL_PATH` inside `src/config.py`.

## 4. Quick Sanity Check

Check that the dataset loads:

```bash
python -m src.run_check
```

Expected output (example):
```
[info] checking setup...
[info] expecting JSONL at: /full/path/data/raw/resumes_dataset.jsonl
[info] loaded JSONL dataset from /full/path/data/raw/resumes_dataset.jsonl
[info] shape: (1234, 3)
[info] columns: ['id', 'text', 'title']
[success] dataset loaded.
```

If you see `FileNotFoundError`, confirm the path and filename.

## 5. Planned Scripts (Coming Soon)

| Script | Purpose |
|--------|---------|
| `src/preprocess.py` | Clean raw text (PII removal, normalization, title mapping) and write processed CSV to `data/processed/` |
| `src/train_baseline.py` | Train TF-IDF + LogisticRegression baseline model |
| `src/evaluate.py` | Compute accuracy, F1, confusion matrix, top-k metrics |
| `src/train_transformer.py` | Fine-tune a transformer (e.g., DistilBERT) on title classification |

## 6. Running Baselines (Example Flow)

```bash
# (After you implement preprocess.py)
python src/preprocess.py \
	--in data/raw/resumes_dataset.jsonl \
	--out data/processed/resumes_processed.csv

# Train baseline (to be added)
python src/train_baseline.py \
	--data data/processed/resumes_processed.csv \
	--model-dir models/baseline

# Evaluate (to be added)
python src/evaluate.py \
	--data data/processed/resumes_processed.csv \
	--model-dir models/baseline
```

## 7. Configuration

Key paths are centralized in `src/config.py`. Adjust `RAW_JSONL_PATH` there if you move the file. Avoid hardcoding paths in other scripts; import from `config.py` instead:

```python
from src.config import RAW_JSONL_PATH
```

## 8. Common Issues & Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'src'` | Running script directly (`python src/run_check.py`) | Use module form: `python -m src.run_check` or set `PYTHONPATH=.` |
| `FileNotFoundError` for dataset | Wrong filename/path | Ensure JSONL in `data/raw/` and matches `RAW_JSONL_PATH` |
| Missing package (ImportError) | Dependency not installed | Re-run: `pip install -r requirements.txt` |
| Memory spike loading JSONL | Very large file | Consider streaming line-by-line or chunking with pandas `chunksize=` |

## 9. Development Conventions

- Use `python -m` to run modules to keep imports consistent.
- Keep raw data immutable; produce derived artifacts in `data/processed/`.
- Log essential metadata (shape, columns) in each script.
- Prefer deterministic runs; set `RANDOM_SEED` for splits later.

## 10. Next Steps

1. Implement `preprocess.py` (text cleaning, title normalization).
2. Add baseline training script and save model artifacts under `models/`.
3. Add evaluation metrics script.
4. Extend README with actual command-line arguments once scripts exist.
5. For progress report: export sample rows, counts, and early baseline metrics.

## 11. License / Attribution

If using a public dataset (e.g., Kaggle), cite original source in your report and add links in a future `SOURCES.md`.

---
Questions or improvements? Open an issue or propose a PR.
