# CareerCompass: Deep Technical Investigation Report

**Date:** December 2024  
**Purpose:** Comprehensive implementation analysis for final report writing

---

## 1. Repository Structure Overview

### Core Python Modules

#### Configuration
- **`src/config.py`**
  - Purpose: Centralized path and constant definitions
  - Key constants:
    - `PROJECT_ROOT`: Directory above `src/`
    - `RANDOM_SEED = 42` (used for all random operations)
    - Paths: `DATA_DIR`, `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `FEATURES_DIR`, `MODELS_DIR`, `RUNS_DIR`
    - Canonical paths:
      - `RAW_JSONL_PATH = data/raw/resumes_dataset.jsonl`
      - `PROCESSED_PARQUET = data/processed/resumes_v1.parquet`
      - `SPLITS_JSON = data/processed/splits_v1.json`

#### Data Pipeline Modules
- **`src/load_dataset.py`**
  - Function: `load_jsonl_dataset(path=None) -> pd.DataFrame`
  - Purpose: Load raw JSONL resume data into pandas DataFrame
  - CLI entrypoint: `python -m src.load_dataset`

- **`src/data_filter.py`**
  - Function: `load_and_filter() -> pd.DataFrame`
  - Purpose: Filter resumes that have at least one of: `["Summary", "Experience", "Education", "Skills", "Text"]`
  - IMPORTANT_FIELDS: `["Summary", "Experience", "Education", "Skills", "Text"]`
  - CLI entrypoint: `python -m src.data_filter`

- **`src/text_processing.py`**
  - Functions:
    - `scrub_pii(text: str) -> str`: Replace emails with `[EMAIL]`, phones with `[PHONE]`
    - `scrub_columns(df, cols=None) -> pd.DataFrame`: Apply PII scrubbing to columns
    - `add_text_clean(df) -> pd.DataFrame`: Join TEXT_COLUMNS with `\n`, then scrub PII
  - TEXT_COLUMNS: `["Summary", "Experience", "Education", "Skills", "Text"]`
  - Regex patterns:
    - EMAIL_RE: `r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"`
    - PHONE_RE: `r"(\+?\d[\d\s\-\(\)]{6,}\d)"`

- **`src/label_creation.py`**
  - Functions:
    - `normalize_category(raw_cat: str) -> str`: Normalize raw category labels
    - `to_family(clean_label: str) -> str`: Map clean label to occupation family
    - `add_label_columns(df) -> pd.DataFrame`: Add `y_title` and `y_family` columns
  - CATEGORY_NORMALIZATION: Partial mapping (only 5 entries: "software engineer", "java developer", "data scientist", "accountant", "project manager")
  - FAMILY_LOOKUP: Partial mapping (only 5 entries)
  - **Status:** PARTIAL - Most categories fall back to original stripped value

- **`build_clean_dataset.py`**
  - Purpose: Orchestrates entire preprocessing pipeline
  - Steps:
    1. `load_and_filter()` - Load and filter raw data
    2. `scrub_columns(df)` - Scrub PII from text columns
    3. `add_text_clean(df)` - Create combined `text_clean` column
    4. Create `text_norm` = `text_clean.strip()`
    5. Filter rows with empty `text_norm`
    6. `add_label_columns(df)` - Add label columns
    7. Set `title_raw = y_title`
    8. `_assign_resume_ids(df)` - Generate resume_id column
  - Outputs:
    - `data/processed/resumes_v1.parquet` (canonical)
    - `data/processed/resumes_clean.csv` (fallback)
  - CLI entrypoint: `python build_clean_dataset.py`

#### Splits Management
- **`src/utils/splits.py`**
  - Functions:
    - `load_splits(path=None) -> SplitDict | None`: Load existing splits JSON
    - `save_splits(splits, path=None) -> Path`: Persist splits to disk
    - `ensure_splits(df, label_col, id_col, splits_path=None, seed=42) -> SplitDict`: Create stratified splits if missing
  - Split strategy:
    - Test size: 10% (0.1)
    - Val size: 10% (0.1)
    - Train size: 80% (remaining)
    - Uses `train_test_split` with `stratify=label_col` (falls back to no stratification if ValueError)
    - Random seed: 42 (from `config.RANDOM_SEED`)
  - Split format: `{"train": [id1, id2, ...], "val": [...], "test": [...]}`

#### Feature Engineering
- **`src/features/tfidf_build.py`**
  - Function: `build_tfidf_features(...) -> TfidfArtifacts`
  - Purpose: Build TF-IDF features with train-only fitting (prevents data leakage)
  - TF-IDF Parameters (hardcoded):
    - `ngram_range=(1, 2)` - Unigrams and bigrams
    - `min_df=3` - Minimum document frequency
    - `max_df=0.9` - Maximum document frequency (90% of documents)
    - `strip_accents="unicode"`
  - Process:
    1. Load parquet and ensure splits exist
    2. Filter to training rows only
    3. Fit `TfidfVectorizer` on training text only
    4. Transform all data (train + val + test)
    5. Save artifacts:
       - `tfidf_X.npz` - Sparse matrix (scipy.sparse format)
       - `tfidf_index.parquet` - DataFrame mapping row indices to resume_id and split
       - `tfidf_vectorizer.joblib` - Fitted vectorizer
  - Output directory: `data/features/` (default: `FEATURES_DIR`)
  - CLI entrypoint: `python -m src.features.tfidf_build`
  - **Actual TF-IDF Stats:**
    - Matrix shape: (3500, 79991)
    - Vocabulary size: 79,991 features
    - Sparsity: 99.45%
    - Index rows: 3,500

#### Model Implementations

##### Majority Baseline
- **`src/models/majority.py`**
  - Function: `majority_label(series: pd.Series) -> str`
  - Purpose: Always predict the most frequent training label
  - Process:
    1. Load data and splits
    2. Find most common label in training set
    3. Predict that label for all val/test samples
    4. Compute accuracy and macro-F1
  - Metrics computed: Accuracy, Macro-F1
  - Output: `runs/majority_baseline/metrics.json`
  - CLI entrypoint: `python -m src.models.majority`

##### SVM Baseline
- **`src/models/train_svm.py`**
  - Functions:
    - `_maybe_build_tfidf(args)`: Build TF-IDF if artifacts missing
    - `_load_artifacts(tfidf_dir)`: Load TF-IDF matrix and index
    - `_prepare_labels(...)`: Load and align labels with TF-IDF matrix rows
    - `_evaluate_split(...)`: Evaluate model on a split, generate confusion matrix
    - `_train_model(X_train, y_train, C) -> LinearSVC`: Train LinearSVC with specified C
    - `parse_args()`: CLI argument parser
    - `main()`: Orchestrates training and evaluation
  - Model: `sklearn.svm.LinearSVC`
  - Hyperparameters:
    - `C`: Tuned via grid search if `--tune` flag used
    - Grid: `[0.25, 0.5, 1.0, 2.0]` (if tuning) or `[1.0]` (default)
    - `class_weight="balanced"` - Handles class imbalance
    - `dual=True` - Use dual formulation
    - `max_iter=5000` - Maximum iterations
    - `random_state=42` - From `config.RANDOM_SEED`
  - Loss function: Hinge loss (soft-margin SVM, λ = 1/C)
  - Optimizer: LibLinear (coordinate descent)
  - Label encoding: `sklearn.preprocessing.LabelEncoder` (fitted on training labels)
  - Metrics computed:
    - Accuracy
    - Macro-F1
    - Top-1 accuracy (from decision_function scores)
    - Top-3 accuracy (from decision_function scores)
  - Artifacts saved:
    - `models/svm_title.joblib` - Trained model
    - `models/label_encoder.joblib` - Label encoder (shared with other models)
    - `runs/svm_tfidf/metrics.json` - Validation and test metrics
    - `runs/svm_tfidf/confusion_val.png` - Validation confusion matrix
    - `runs/svm_tfidf/confusion_test.png` - Test confusion matrix
  - CLI entrypoint: `python -m src.models.train_svm [--tune]`
  - **Actual Training Results:**
    - Best C: 2.0 (from grid search)
    - Validation: Accuracy=0.9000, Macro-F1=0.9357, Top-1=0.9000, Top-3=0.9771
    - Test: Accuracy=0.8857, Macro-F1=0.9242, Top-1=0.8857, Top-3=0.9743

##### BERT Model
- **`src/models/train_bert.py`**
  - Classes:
    - `ResumeDataset(torch.utils.data.Dataset)`: PyTorch dataset for tokenized inputs
      - `__getitem__`: Tokenizes text with truncation, max_length padding
      - Returns: `{"input_ids": tensor, "attention_mask": tensor, "labels": tensor}`
  - Functions:
    - `load_or_fit_label_encoder(df_train, label_col) -> LabelEncoder`: Loads or fits encoder, saves to `models/label_encoder.joblib`
    - `compute_metrics(pred) -> Dict`: Computes accuracy and macro-F1 for Hugging Face Trainer
    - `parse_args() -> argparse.Namespace`: CLI argument parser
    - `main()`: Orchestrates BERT training
  - Model: `transformers.BertForSequenceClassification`
  - Base model: `bert-base-uncased` (default, can be changed via `--model-name`)
  - Hyperparameters (defaults):
    - `--epochs`: 3
    - `--batch-size`: 8 (per device)
    - `--eval-batch-size`: 16 (per device)
    - `--learning-rate`: 2e-5
    - `--max-length`: 512 (tokenization)
  - TrainingArguments (hardcoded):
    - `num_train_epochs`: From `--epochs` arg
    - `per_device_train_batch_size`: From `--batch-size` arg
    - `per_device_eval_batch_size`: From `--eval-batch-size` arg
    - `learning_rate`: From `--learning-rate` arg
    - `weight_decay`: 0.01
    - `evaluation_strategy`: "epoch"
    - `save_strategy`: "epoch"
    - `load_best_model_at_end`: True
    - `metric_for_best_model`: "macro_f1"
    - `seed`: 42 (from `config.RANDOM_SEED`)
    - `fp16`: True if CUDA available, else False
    - `logging_steps`: 50
    - `save_total_limit`: 2 (keeps only 2 checkpoints)
  - Loss function: Cross-entropy (softmax output layer)
  - Optimizer: AdamW (default from Hugging Face Trainer)
  - Label encoding: Shared `LabelEncoder` from `models/label_encoder.joblib`
  - Model loading strategy:
    1. If `--local-model-dir` provided: Load from local directory with `local_files_only=True`
    2. Else: Try cache first with `local_files_only=True`, then online download
  - Metrics computed:
    - Accuracy (via `compute_metrics` function)
    - Macro-F1 (via `compute_metrics` function)
  - Artifacts saved:
    - `models/bert_title/` - Model weights, config, tokenizer files
    - `runs/bert_title/metrics.json` - Validation and test metrics
    - `runs/bert_title/confusion_val.png` - Validation confusion matrix
    - `runs/bert_title/confusion_test.png` - Test confusion matrix
    - `runs/bert_title/checkpoint-*` - Training checkpoints (up to 2 kept)
  - CLI entrypoint: `python -m src.models.train_bert [--epochs N] [--batch-size N] [--eval-batch-size N] [--learning-rate F] [--local-model-dir PATH]`
  - **Training Status:** NOT COMPLETED
    - `models/bert_title/`: Empty directory (created but no files)
    - `runs/bert_title/`: Empty directory (created but no files)
    - No metrics.json, no confusion matrices, no checkpoints

- **`src/models/inference_bert.py`**
  - Functions:
    - `load_bert_model(model_dir=None) -> Tuple`: Loads trained BERT model, tokenizer, label encoder
    - `predict_bert(texts, model, tokenizer, label_encoder, device, max_length=512, batch_size=16) -> Dict`: Performs inference on raw texts
  - Purpose: Programmatic API for BERT inference
  - CLI entrypoint: `python -m src.models.inference_bert <text1> [text2] ...`

#### Evaluation Modules
- **`src/eval/metrics.py`**
  - Functions:
    - `accuracy(y_true, y_pred) -> float`: Simple accuracy calculation
    - `macro_f1(y_true, y_pred) -> float`: Macro-averaged F1 score
    - `confusion(y_true, y_pred, labels, display_labels=None, scale=1.25) -> tuple`: Generate confusion matrix plot
    - `topk_from_scores(y_true, scores, k) -> float`: Top-k accuracy from decision function scores
  - Used by: `train_svm.py`, `train_bert.py`

- **`src/eval/compare_models.py`**
  - Function: `compare_models(...) -> Dict`
  - Purpose: Load and compare metrics from Majority, SVM, and BERT models
  - Process:
    1. Load metrics JSON files from each model's run directory
    2. Build comparison structure with validation and test metrics
    3. Save to `runs/comparison/model_comparison.json`
    4. Print formatted table to console
  - Handles missing BERT gracefully (marks as N/A)
  - Output: `runs/comparison/model_comparison.json`
  - CLI entrypoint: `python -m src.eval.compare_models`
  - **Current Output:**
    - Majority: Present, metrics loaded
    - SVM: Present, metrics loaded
    - BERT: Not present (metrics.json missing)

- **`src/eval/error_analysis.py`**
  - Function: `analyze_errors(split="test", output_dir=None, include_bert=True) -> pd.DataFrame`
  - Purpose: Compare predictions across models and identify error patterns
  - Process:
    1. Load SVM model, vectorizer, label encoder
    2. Generate predictions for SVM and Majority
    3. Optionally load BERT and generate predictions (gracefully skips if unavailable)
    4. Categorize errors:
       - `both_wrong`: SVM and Majority both incorrect
       - `svm_wrong_bert_right`: SVM wrong, BERT correct (if BERT available)
       - `svm_right_bert_wrong`: SVM correct, BERT wrong (if BERT available)
       - `svm_bert_both_wrong`: Both SVM and BERT wrong (if BERT available)
    5. Export to CSV with columns: resume_id, true_label, svm_pred, majority_pred, bert_pred (if available), correctness flags, text_snippet
  - Output: `runs/comparison/error_cases_{split}.csv`
  - CLI entrypoint: `python -m src.eval.error_analysis [--split val|test] [--no-bert]`
  - **Current Outputs:**
    - `runs/comparison/error_cases_val.csv`: 350 rows
    - `runs/comparison/error_cases_test.csv`: 350 rows
    - BERT predictions: Not included (model not trained)

#### Utility Modules
- **`src/setup_env.py`**: Environment setup checker
- **`src/run_check.py`**: Quick sanity check for raw data loading

---

## 2. Data Pipeline (Actual Implementation)

### Raw Data
- **Source:** `data/raw/resumes_dataset.jsonl`
- **Format:** JSON Lines (one JSON object per line)
- **Loading:** `pandas.read_json(path, lines=True)`

### Filtering
- **Criteria:** Keep rows where at least one of `["Summary", "Experience", "Education", "Skills", "Text"]` is not null
- **Implementation:** `df[IMPORTANT_FIELDS].notna().any(axis=1)`

### Text Processing
- **PII Scrubbing:**
  - Emails → `[EMAIL]`
  - Phone numbers → `[PHONE]`
  - Applied to individual columns before joining
- **Text Concatenation:**
  - Join `["Summary", "Experience", "Education", "Skills", "Text"]` with `\n`
  - Result stored in `text_clean` column
  - PII scrubbing applied again after joining
- **Normalization:**
  - `text_norm = text_clean.strip()`
  - Rows with empty `text_norm` are filtered out
- **Text Statistics:**
  - Mean length: 4,164 characters
  - Median length: 2,844 characters

### Label Creation
- **Source column:** `Category` (from raw JSONL)
- **Normalization:** Partial mapping in `CATEGORY_NORMALIZATION` dict (only 5 entries)
  - Most categories fall back to original stripped value
- **Output columns:**
  - `y_title`: Normalized category (used as `title_raw` in final dataset)
  - `y_family`: Occupation family (partial mapping, most fall back to "Other")
- **Label Statistics:**
  - Total samples: 3,500
  - Unique labels: 36
  - Top labels:
    - Java Developer: 200
    - Data Science: 200
    - Python Developer: 200
    - DevOps: 180
    - SQL Developer: 180
    - Web Designing: 150
    - Business Analyst: 150
    - React Developer: 150
    - Testing: 150
    - Database: 150

### Splits Creation
- **File:** `data/processed/splits_v1.json`
- **Strategy:** Stratified train/val/test split
- **Proportions:** 80% train, 10% val, 10% test
- **Random seed:** 42
- **Actual splits:**
  - Train: 2,800 samples
  - Val: 350 samples
  - Test: 350 samples
- **Creation:** `ensure_splits()` function in `src/utils/splits.py`
  - Uses `sklearn.model_selection.train_test_split` with `stratify=label_col`
  - Falls back to no stratification if ValueError (e.g., single sample per class)

### Final Processed Dataset
- **File:** `data/processed/resumes_v1.parquet`
- **Shape:** (3500, N columns)
- **Key columns:**
  - `resume_id`: Unique identifier (assigned by `_assign_resume_ids()`)
  - `text_clean`: Concatenated and PII-scrubbed text
  - `text_norm`: Normalized text (used for TF-IDF and BERT)
  - `title_raw`: Job title label (from `y_title`)
  - `y_family`: Occupation family label
- **Fallback:** `data/processed/resumes_clean.csv` (same data, CSV format)

---

## 3. Feature Engineering

### TF-IDF Features
- **Implementation:** `src/features/tfidf_build.py`
- **Vectorizer:** `sklearn.feature_extraction.text.TfidfVectorizer`
- **Parameters:**
  - `ngram_range=(1, 2)` - Unigrams and bigrams
  - `min_df=3` - Minimum document frequency (term must appear in at least 3 documents)
  - `max_df=0.9` - Maximum document frequency (term must appear in at most 90% of documents)
  - `strip_accents="unicode"` - Remove accents
- **Fitting strategy:** Fit only on training data (prevents data leakage)
- **Matrix properties:**
  - Shape: (3500, 79991)
  - Vocabulary size: 79,991 features
  - Sparsity: 99.45%
  - Format: Scipy sparse matrix (CSR format, saved as NPZ)
- **Artifacts:**
  - `data/features/tfidf_X.npz` - Sparse feature matrix
  - `data/features/tfidf_index.parquet` - Row index mapping (resume_id, row_ix, split)
  - `data/features/tfidf_vectorizer.joblib` - Fitted vectorizer
- **Usage:** Loaded by `train_svm.py` for SVM training

### Other Features
- **`data/features/transformer_embeddings.npy`**: Present but unused (no code references it)
- **No other feature engineering modules found**

---

## 4. Models Implemented

### 4.1 Majority Baseline

**Location:** `src/models/majority.py`

**Implementation:**
- Finds most frequent label in training set
- Predicts that label for all validation and test samples

**Metrics:**
- Validation: Accuracy=0.0571, Macro-F1=0.0030
- Test: Accuracy=0.0571, Macro-F1=0.0030

**Artifacts:**
- `runs/majority_baseline/metrics.json`

**Status:** ✅ COMPLETE

---

### 4.2 SVM Baseline (LinearSVC)

**Location:** `src/models/train_svm.py`

**Model:** `sklearn.svm.LinearSVC`

**Hyperparameters:**
- `C`: Tuned via grid search over `[0.25, 0.5, 1.0, 2.0]` (if `--tune` flag used)
- **Best C:** 2.0 (selected based on validation macro-F1)
- `class_weight="balanced"` - Handles class imbalance
- `dual=True` - Dual formulation
- `max_iter=5000` - Maximum solver iterations
- `random_state=42` - Reproducibility

**Loss Function:**
- Soft-margin hinge loss: `min_{w,b} λ||w||² + (1/N)∑max(0, 1 - y_i(w·x_i + b))`
- Where λ = 1/C (scikit-learn parameterization)
- C = 2.0 → λ = 0.5

**Optimizer:**
- LibLinear (coordinate descent)
- Deterministic convex solver

**Label Encoding:**
- `sklearn.preprocessing.LabelEncoder`
- Fitted on training labels
- Saved to `models/label_encoder.joblib` (shared with BERT)

**Training Process:**
1. Load or build TF-IDF features
2. Align labels with TF-IDF matrix rows
3. Split into train/val/test based on `tfidf_index.parquet`
4. Grid search over C values (if `--tune` flag)
5. Select best model based on validation macro-F1
6. Evaluate on validation and test sets
7. Generate confusion matrices
8. Save model, encoder, and metrics

**Metrics Computed:**
- Accuracy
- Macro-F1
- Top-1 accuracy (from `decision_function` scores)
- Top-3 accuracy (from `decision_function` scores)

**Artifacts:**
- `models/svm_title.joblib` - Trained model
- `models/label_encoder.joblib` - Label encoder
- `runs/svm_tfidf/metrics.json` - Metrics
- `runs/svm_tfidf/confusion_val.png` - Validation confusion matrix
- `runs/svm_tfidf/confusion_test.png` - Test confusion matrix

**Results:**
- Validation: Accuracy=0.9000, Macro-F1=0.9357, Top-1=0.9000, Top-3=0.9771
- Test: Accuracy=0.8857, Macro-F1=0.9242, Top-1=0.8857, Top-3=0.9743

**Status:** ✅ COMPLETE

---

### 4.3 BERT Model

**Location:** `src/models/train_bert.py`

**Model:** `transformers.BertForSequenceClassification`

**Base Model:** `bert-base-uncased` (default, 110M parameters)

**Hyperparameters (defaults):**
- Epochs: 3
- Batch size (train): 8 per device
- Batch size (eval): 16 per device
- Learning rate: 2e-5
- Max sequence length: 512 tokens
- Weight decay: 0.01 (hardcoded)

**TrainingArguments (Hugging Face):**
- `evaluation_strategy="epoch"` - Evaluate after each epoch
- `save_strategy="epoch"` - Save checkpoint after each epoch
- `load_best_model_at_end=True` - Load best model based on metric
- `metric_for_best_model="macro_f1"` - Select best by macro-F1
- `fp16=True` if CUDA available, else `False`
- `logging_steps=50` - Log every 50 steps
- `save_total_limit=2` - Keep only 2 checkpoints
- `seed=42` - Random seed

**Loss Function:**
- Cross-entropy loss (softmax output layer)

**Optimizer:**
- AdamW (default from Hugging Face Trainer)
- Learning rate: 2e-5
- Weight decay: 0.01

**Label Encoding:**
- Shared `LabelEncoder` from `models/label_encoder.joblib`
- Same encoding as SVM (ensures consistency)

**Dataset:**
- `ResumeDataset` class (PyTorch Dataset)
- Tokenization: BERT tokenizer with truncation and max_length padding
- Max length: 512 tokens

**Model Loading:**
1. If `--local-model-dir` provided: Load from local directory
2. Else: Try cache first, then download from Hugging Face Hub
3. Handles network errors gracefully with clear error messages

**Metrics Computed:**
- Accuracy (via `compute_metrics` function)
- Macro-F1 (via `compute_metrics` function)

**Artifacts (expected):**
- `models/bert_title/` - Model weights, config, tokenizer
- `runs/bert_title/metrics.json` - Metrics
- `runs/bert_title/confusion_val.png` - Validation confusion matrix
- `runs/bert_title/confusion_test.png` - Test confusion matrix
- `runs/bert_title/checkpoint-*` - Training checkpoints

**Status:** ❌ NOT TRAINED
- `models/bert_title/`: Empty directory (created Dec 4 22:01)
- `runs/bert_title/`: Empty directory (created Dec 4 22:01)
- No metrics, no confusion matrices, no checkpoints
- Training attempted but not completed (likely network/download issues)

---

## 5. Training Status

### SVM Training
- **Status:** ✅ COMPLETE
- **Run directory:** `runs/svm_tfidf/`
- **Files present:**
  - `metrics.json` - Contains validation and test metrics
  - `confusion_val.png` - Validation confusion matrix
  - `confusion_test.png` - Test confusion matrix
- **Model saved:** `models/svm_title.joblib`
- **Label encoder saved:** `models/label_encoder.joblib`

### Majority Baseline
- **Status:** ✅ COMPLETE
- **Run directory:** `runs/majority_baseline/`
- **Files present:**
  - `metrics.json` - Contains validation and test metrics

### BERT Training
- **Status:** ❌ NOT COMPLETED
- **Run directory:** `runs/bert_title/` - Empty
- **Model directory:** `models/bert_title/` - Empty
- **Evidence of attempt:**
  - Directories created on Dec 4 22:01
  - No model files, no metrics, no checkpoints
- **Possible reasons:**
  - Network issues downloading model from Hugging Face
  - Training interrupted before completion
  - Model download failed (tokenizer files may be in cache, but model weights missing)

---

## 6. Evaluation System

### 6.1 Model Comparison

**Script:** `src/eval/compare_models.py`

**Functionality:**
- Loads metrics JSON files from:
  - `runs/majority_baseline/metrics.json`
  - `runs/svm_tfidf/metrics.json`
  - `runs/bert_title/metrics.json` (optional)
- Builds comparison structure with validation and test metrics
- Handles missing BERT gracefully (marks as N/A)
- Saves to `runs/comparison/model_comparison.json`
- Prints formatted table to console

**Current Output:** `runs/comparison/model_comparison.json`
- Majority: Present, metrics loaded
- SVM: Present, metrics loaded
- BERT: Not present (metrics.json missing)

**CLI:** `python -m src.eval.compare_models`

---

### 6.2 Error Analysis

**Script:** `src/eval/error_analysis.py`

**Functionality:**
- Loads SVM model, vectorizer, and label encoder
- Generates predictions for SVM and Majority baselines
- Optionally loads BERT and generates predictions (gracefully skips if unavailable)
- Categorizes errors:
  - `both_wrong`: SVM and Majority both incorrect
  - `svm_wrong_bert_right`: SVM wrong, BERT correct (if BERT available)
  - `svm_right_bert_wrong`: SVM correct, BERT wrong (if BERT available)
  - `svm_bert_both_wrong`: Both SVM and BERT wrong (if BERT available)
- Exports to CSV with columns:
  - `resume_id`, `true_label`, `svm_pred`, `majority_pred`, `bert_pred` (if available)
  - `svm_correct`, `majority_correct`, `bert_correct` (if available)
  - `both_wrong`, `svm_wrong_bert_right`, `svm_right_bert_wrong`, `svm_bert_both_wrong` (if BERT available)
  - `text_snippet` (first 200 characters)

**Current Outputs:**
- `runs/comparison/error_cases_val.csv`: 350 rows
- `runs/comparison/error_cases_test.csv`: 350 rows
- BERT predictions: Not included (model not trained)

**Error Statistics (from test set):**
- Total samples: 350
- SVM correct: 310 (88.6%)
- SVM wrong: 40 (11.4%)
- Majority correct: 20 (5.7%)
- Majority wrong: 330 (94.3%)
- Hard cases (both wrong): 37

**CLI:** `python -m src.eval.error_analysis [--split val|test] [--no-bert]`

---

## 7. Training Environment Notes

### Current Environment
- **OS:** macOS (darwin 24.6.0)
- **Python:** 3.13 (based on paths)
- **PyTorch:** 2.9.0
- **Transformers:** 4.57.1
- **CUDA:** Not available (CPU training only)

### BERT Training Attempts
- Directories created: Dec 4 22:01
- No successful training completion
- Likely issues:
  - Network connectivity for model download
  - Model weights not fully cached
  - Training interrupted before completion

### No Evidence of Training on Other Machines
- No checkpoints from different paths
- No version mismatches in saved artifacts
- All artifacts consistent with local environment

---

## 8. Additional Implementation Details

### Ablations Actually Run
- **SVM C hyperparameter tuning:** Yes (grid search over [0.25, 0.5, 1.0, 2.0])
- **TF-IDF variants:** No (single configuration used)
- **BERT batch size variations:** No (defaults used, training not completed)
- **BERT learning rate variations:** No (default 2e-5 used, training not completed)

### Intended vs. Actual Design
- **Label normalization:** Intended to normalize all categories, but only 5 entries in mapping (most fall back to original)
- **BERT training:** Intended to complete, but not executed successfully
- **Error analysis with BERT:** Intended to include BERT, but gracefully handles missing model

### TODO Comments
- **None found** (grep for TODO/FIXME/XXX/HACK returned no results)

### Incomplete Features
- **BERT training:** Not completed
- **Label normalization:** Partial (only 5 categories mapped)
- **Family mapping:** Partial (only 5 families mapped)
- **Transformer embeddings:** File exists (`transformer_embeddings.npy`) but unused

### Final Performance Numbers

**Majority Baseline:**
- Validation: Accuracy=0.0571, Macro-F1=0.0030
- Test: Accuracy=0.0571, Macro-F1=0.0030

**SVM Baseline:**
- Validation: Accuracy=0.9000, Macro-F1=0.9357, Top-1=0.9000, Top-3=0.9771
- Test: Accuracy=0.8857, Macro-F1=0.9242, Top-1=0.8857, Top-3=0.9743
- Best C: 2.0

**BERT:**
- Not available (training not completed)

---

## 9. File Artifacts Summary

### Data Files
- `data/raw/resumes_dataset.jsonl` - Raw input data
- `data/processed/resumes_v1.parquet` - Processed dataset (canonical)
- `data/processed/resumes_clean.csv` - Processed dataset (CSV fallback)
- `data/processed/splits_v1.json` - Train/val/test splits

### Feature Files
- `data/features/tfidf_X.npz` - TF-IDF sparse matrix
- `data/features/tfidf_index.parquet` - Row index mapping
- `data/features/tfidf_vectorizer.joblib` - Fitted vectorizer
- `data/features/transformer_embeddings.npy` - Unused embeddings file

### Model Files
- `models/svm_title.joblib` - Trained SVM model
- `models/label_encoder.joblib` - Shared label encoder
- `models/bert_title/` - Empty (BERT not trained)

### Run Artifacts
- `runs/majority_baseline/metrics.json` - Majority metrics
- `runs/svm_tfidf/metrics.json` - SVM metrics
- `runs/svm_tfidf/confusion_val.png` - SVM validation confusion matrix
- `runs/svm_tfidf/confusion_test.png` - SVM test confusion matrix
- `runs/bert_title/` - Empty (BERT not trained)
- `runs/comparison/model_comparison.json` - Model comparison
- `runs/comparison/error_cases_val.csv` - Validation error analysis
- `runs/comparison/error_cases_test.csv` - Test error analysis

---

## 10. Dependencies

**requirements.txt:**
```
pandas
numpy
scikit-learn
scipy
joblib
pyarrow
matplotlib
torch
transformers
```

**Key Versions (from environment):**
- PyTorch: 2.9.0
- Transformers: 4.57.1
- scikit-learn: (version not checked, but used for LinearSVC, LabelEncoder, train_test_split)
- pandas: (used for data loading and manipulation)
- numpy: (used for array operations)

---

**END OF INVESTIGATION REPORT**

