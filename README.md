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
        data_filter.py         <- Data filtering utilities
        label_creation.py      <- Label creation utilities
        load_dataset.py        <- Loads JSONL into a pandas DataFrame
        run_check.py           <- Quick environment & dataset existence check
        setup_env.py           <- Prints versions of key libs
        text_prcoessing.py     <- Text processing utilities
    build_clean_dataset.py    <- Script to create processed dataset
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

## 5. Creating processed (cleaner) dataset

Firstly, check that you have a "processed" folder under the "data folder"

```
CareerCompass/
	README.md           
    build_clean_dataset.py
	requirements.txt        
	data/
		raw/       
		processed/      <----- target folder
	src/
		config.py        
        data_filter.py
        label_creation.py   
		load_dataset.py   
		run_check.py      
		setup_env.py   
        text_prcoessing.py

	docs/                    

```
Once you ensure that there is a "processed" folder, run the following command:

```bash
python build_clean_dataset.py
```
