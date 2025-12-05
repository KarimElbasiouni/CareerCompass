# CareerCompass: Evaluation Strategy & Results Investigation

**Date:** December 2024  
**Purpose:** Comprehensive evaluation analysis for final report writing

---

## 1. Dataset Split Details

### Split Sizes
- **Train:** 2,800 samples (80%)
- **Validation:** 350 samples (10%)
- **Test:** 350 samples (10%)
- **Total:** 3,500 samples

### Stratification Status
- **Stratified:** Yes (attempted via `sklearn.model_selection.train_test_split` with `stratify=label_col`)
- **Fallback:** If stratification fails (ValueError), falls back to non-stratified split
- **Random seed:** 42 (from `config.RANDOM_SEED`)

### Label Distribution Per Split

**Total unique labels:** 36 classes

**Training set:**
- Unique labels: 36
- Min count per class: 16
- Max count per class: 160
- Classes with < 5 examples: 0

**Validation set:**
- Unique labels: 36
- Min count per class: 2
- Max count per class: 20
- Classes with < 5 examples: 7

**Test set:**
- Unique labels: 36
- Min count per class: 2
- Max count per class: 20
- Classes with < 5 examples: 7

### Detailed Label Counts (All 36 Classes)

| Label | Train | Val | Test |
|-------|-------|-----|------|
| AI Engineer | 57 | 7 | 7 |
| Backend Developer | 61 | 7 | 8 |
| Blockchain | 37 | 5 | 5 |
| Blockchain Developer | 24 | 3 | 3 |
| Business Analyst | 120 | 15 | 15 |
| Cloud Engineer | 74 | 9 | 9 |
| Cybersecurity Analyst | 52 | 7 | 7 |
| Data Science | 160 | 20 | 20 |
| Database | 120 | 15 | 15 |
| Database Administrator | 44 | 6 | 6 |
| DevOps | 144 | 18 | 18 |
| Digital Media | 80 | 10 | 10 |
| DotNet Developer | 112 | 14 | 14 |
| ETL Developer | 96 | 12 | 12 |
| Engineering Manager | 24 | 3 | 3 |
| Frontend Developer | 61 | 7 | 8 |
| Full Stack Developer | 82 | 10 | 10 |
| Java Developer | 160 | 20 | 20 |
| Machine Learning Engineer | 65 | 8 | 8 |
| Mobile Developer | 36 | 5 | 5 |
| Network Security Engineer | 96 | 12 | 12 |
| Principal Engineer | 20 | 3 | 2 |
| Product Manager | 21 | 2 | 2 |
| Python Developer | 160 | 20 | 20 |
| QA Engineer | 49 | 6 | 6 |
| React Developer | 120 | 15 | 15 |
| SAP Developer | 80 | 10 | 10 |
| SQL Developer | 144 | 18 | 18 |
| Site Reliability Engineer | 36 | 5 | 5 |
| Software Developer | 108 | 13 | 13 |
| System Administrator | 32 | 4 | 4 |
| Technical Lead | 28 | 4 | 3 |
| Technical Writer | 16 | 2 | 2 |
| Testing | 120 | 15 | 15 |
| UI/UX Designer | 41 | 5 | 5 |
| Web Designing | 120 | 15 | 15 |

### Class Imbalance Observations
- **Most frequent classes:** Java Developer, Data Science, Python Developer (160 train samples each)
- **Least frequent classes:** Technical Writer (16 train samples), Principal Engineer (20), Product Manager (21)
- **Imbalance ratio:** ~10:1 (160 max / 16 min in training)
- **Small classes in val/test:** 7 classes have < 5 examples in validation and test sets
  - These include: Blockchain Developer (3), Engineering Manager (3), Principal Engineer (2-3), Product Manager (2), Technical Writer (2)

---

## 2. Evaluation Metrics Used

### Metrics Computed by Each Model

#### Majority Baseline
- **Accuracy** (computed via `sklearn.metrics.accuracy_score`)
- **Macro-F1** (computed via `sklearn.metrics.f1_score` with `average="macro"`, `zero_division=0`)
- **Top-k accuracy:** NOT computed (baseline always predicts same label)
- **Confusion matrices:** NOT generated
- **Per-class metrics:** NOT computed

#### SVM Baseline
- **Accuracy** (computed via `src.eval.metrics.accuracy`)
- **Macro-F1** (computed via `src.eval.metrics.macro_f1`)
- **Top-1 accuracy** (computed via `src.eval.metrics.topk_from_scores` with k=1)
- **Top-3 accuracy** (computed via `src.eval.metrics.topk_from_scores` with k=3)
- **Confusion matrices:** Generated for both validation and test sets
- **Per-class metrics:** NOT explicitly computed (only via confusion matrix inspection)
- **Decision function scores:** Used for Top-k computation (from `model.decision_function(X)`)

#### BERT Model
- **Accuracy** (computed via `src.eval.metrics.accuracy` in `compute_metrics` function)
- **Macro-F1** (computed via `src.eval.metrics.macro_f1` in `compute_metrics` function)
- **Top-k accuracy:** NOT computed (no probability scores extracted for Top-k)
- **Confusion matrices:** Generated for both validation and test sets
- **Per-class metrics:** NOT explicitly computed (only via confusion matrix inspection)
- **Logits/probabilities:** Available from model output but not saved to files

### Metric Implementation Details

**Accuracy:**
- Formula: `(y_true == y_pred).mean()`
- Implementation: `src.eval.metrics.accuracy()`
- Returns 0.0 if empty arrays

**Macro-F1:**
- Formula: Unweighted mean of per-class F1 scores
- Implementation: `sklearn.metrics.f1_score(y_true, y_pred, average="macro")`
- Returns 0.0 if empty arrays

**Top-k Accuracy:**
- Formula: Proportion of samples where true label appears in top-k predictions
- Implementation: `src.eval.metrics.topk_from_scores()`
- Uses `np.argpartition` to find top-k indices from decision function scores
- Only computed for SVM (requires decision function scores)

**Confusion Matrix:**
- Implementation: `sklearn.metrics.confusion_matrix`
- Visualization: Custom function in `src.eval.metrics.confusion()`
- Features:
  - Shows count and row percentage in each cell
  - Figure size: `(base_side * scale, base_side * scale)` where `base_side = max(6, min(1.0 * n_classes, 14))` and `scale = 1.25`
  - DPI: 150
  - Color scheme: Blues colormap
  - Text color: White if count > 0.55 * max_val, else black

---

## 3. Evaluation Outputs for Each Model

### 3.1 Majority Baseline

**Metrics file:** `runs/majority_baseline/metrics.json`

**Validation metrics:**
- Accuracy: 0.0571 (20 correct out of 350)
- Macro-F1: 0.0030

**Test metrics:**
- Accuracy: 0.0571 (20 correct out of 350)
- Macro-F1: 0.0030

**Artifacts:**
- `runs/majority_baseline/metrics.json` ✅ Present
- Confusion matrices: ❌ Not generated
- Predictions file: ❌ Not saved
- Per-class metrics: ❌ Not computed

**Notes:**
- Always predicts "Java Developer" (most frequent training label)
- Identical performance on validation and test (expected for constant predictor)

---

### 3.2 SVM Baseline

**Metrics file:** `runs/svm_tfidf/metrics.json`

**Validation metrics:**
- Accuracy: 0.9000 (315 correct out of 350)
- Macro-F1: 0.9357
- Top-1: 0.9000
- Top-3: 0.9771

**Test metrics:**
- Accuracy: 0.8857 (310 correct out of 350)
- Macro-F1: 0.9242
- Top-1: 0.8857
- Top-3: 0.9743

**Hyperparameters:**
- Best C: 2.0 (selected via grid search over [0.25, 0.5, 1.0, 2.0])

**Artifacts:**
- `runs/svm_tfidf/metrics.json` ✅ Present
- `runs/svm_tfidf/confusion_val.png` ✅ Present (569 KB)
- `runs/svm_tfidf/confusion_test.png` ✅ Present (596 KB)
- `models/svm_title.joblib` ✅ Present (trained model)
- Predictions file: ❌ Not saved (only generated during evaluation)
- Per-class metrics: ❌ Not explicitly saved (only visible in confusion matrices)

**Performance summary:**
- Validation: 315 correct, 35 incorrect (10.0% error rate)
- Test: 310 correct, 40 incorrect (11.4% error rate)
- Generalization gap: 1.4 percentage points (validation to test)
- Top-3 accuracy: 97.4% on test (indicates high confidence in top predictions)

---

### 3.3 BERT Model

**Metrics file:** `runs/bert_finetuned/metrics.json` ⚠️ **NOTE:** Comparison script looks for `runs/bert_title/metrics.json` but actual results are in `runs/bert_finetuned/`

**Validation metrics:**
- Accuracy: 0.8229 (288 correct out of 350, estimated)
- Macro-F1: 0.8881

**Test metrics:**
- Accuracy: 0.8800 (308 correct out of 350, estimated)
- Macro-F1: 0.9238

**Artifacts:**
- `runs/bert_finetuned/metrics.json` ✅ Present
- `runs/bert_finetuned/confusion_val.png` ✅ Present (603 KB)
- `runs/bert_finetuned/confusion_test.png` ✅ Present (600 KB)
- `models/bert_finetuned/` (if model was saved): ⚠️ Location unclear (may be in `models/bert_title/` or `models/bert_finetuned/`)
- Predictions file: ❌ Not saved
- Per-class metrics: ❌ Not explicitly computed
- Top-k accuracy: ❌ Not computed (no probability scores extracted)

**Performance summary:**
- Validation: ~288 correct, ~62 incorrect (17.7% error rate)
- Test: ~308 correct, ~42 incorrect (12.0% error rate)
- Generalization: Test accuracy (0.88) > Validation accuracy (0.823) - unusual but possible with early stopping
- **Comparison to SVM:** BERT test accuracy (0.88) is slightly lower than SVM test accuracy (0.886), but BERT macro-F1 (0.924) matches SVM macro-F1 (0.924)

**Training details (from code):**
- Base model: `bert-base-uncased`
- Epochs: 3 (default)
- Batch size: 8 (train), 16 (eval)
- Learning rate: 2e-5
- Max sequence length: 512 tokens
- Best model selection: Based on validation macro-F1
- Checkpoints: Up to 2 kept (save_total_limit=2)

---

## 4. Confusion Matrix Details

### SVM Confusion Matrices

**Validation confusion matrix:**
- File: `runs/svm_tfidf/confusion_val.png`
- Size: 569 KB
- Classes: 36
- Format: PNG image, 150 DPI
- Features: Count + row percentage in each cell

**Test confusion matrix:**
- File: `runs/svm_tfidf/confusion_test.png`
- Size: 596 KB
- Classes: 36
- Format: PNG image, 150 DPI
- Features: Count + row percentage in each cell

**Notable patterns (from error analysis):**
- Common misclassifications:
  - Java Developer ↔ Python Developer
  - Java Developer ↔ React Developer
  - Python Developer ↔ DevOps
  - Related developer roles show off-diagonal clusters

### BERT Confusion Matrices

**Validation confusion matrix:**
- File: `runs/bert_finetuned/confusion_val.png`
- Size: 603 KB
- Classes: 36
- Format: PNG image, 150 DPI
- Features: Count + row percentage in each cell

**Test confusion matrix:**
- File: `runs/bert_finetuned/confusion_test.png`
- Size: 600 KB
- Classes: 36
- Format: PNG image, 150 DPI
- Features: Count + row percentage in each cell

**Note:** Confusion matrices are image files only; no numeric .npy or .csv files found.

---

## 5. Cross-Model Comparison Evidence

### Model Comparison JSON

**File:** `runs/comparison/model_comparison.json`

**Contents:**
- **Majority:** Present ✅
  - Validation: Accuracy=0.0571, Macro-F1=0.0030
  - Test: Accuracy=0.0571, Macro-F1=0.0030
- **SVM:** Present ✅
  - Validation: Accuracy=0.9000, Macro-F1=0.9357, Top-1=0.9000, Top-3=0.9771
  - Test: Accuracy=0.8857, Macro-F1=0.9242, Top-1=0.8857, Top-3=0.9743
  - Hyperparameters: best_C=2.0
- **BERT:** Present ⚠️ (but path mismatch)
  - Metrics path: `/Users/imranchowdhury/CareerCompass/runs/bert_title/metrics.json` (does not exist)
  - Actual metrics: `runs/bert_finetuned/metrics.json` (exists)
  - Comparison shows: accuracy=null, macro_f1=null (because path mismatch)

**Issue:** Comparison script expects `runs/bert_title/metrics.json` but BERT results are in `runs/bert_finetuned/metrics.json`. This causes BERT metrics to show as null in the comparison file.

---

### Error Analysis CSVs

#### Validation Error Analysis

**File:** `runs/comparison/error_cases_val.csv`

**Total samples:** 350

**SVM performance:**
- Correct: 315 (90.00%)
- Wrong: 35 (10.00%)

**Majority performance:**
- Correct: 20 (5.71%)
- Wrong: 330 (94.29%)

**Hard cases:**
- Both wrong: 34 (9.71%)

**Columns:**
- `resume_id`: Resume identifier
- `true_label`: Actual job title
- `svm_pred`: SVM prediction
- `majority_pred`: Majority baseline prediction
- `svm_correct`: Boolean flag
- `majority_correct`: Boolean flag
- `both_wrong`: Boolean flag (SVM and Majority both incorrect)
- `text_snippet`: First 200 characters of resume text

**BERT predictions:** ❌ Not included (no `bert_pred` or `bert_correct` columns)

**Sample error cases:**
- REAL_0191: True=Java Developer, SVM=React Developer, Majority=Java Developer
- REAL_0203: True=Python Developer, SVM=DevOps, Majority=Java Developer
- REAL_0232: True=Python Developer, SVM=Java Developer, Majority=Java Developer

#### Test Error Analysis

**File:** `runs/comparison/error_cases_test.csv`

**Total samples:** 350

**SVM performance:**
- Correct: 310 (88.57%)
- Wrong: 40 (11.43%)

**Majority performance:**
- Correct: 20 (5.71%)
- Wrong: 330 (94.29%)

**Hard cases:**
- Both wrong: 37 (10.57%)

**Columns:** Same as validation CSV

**BERT predictions:** ❌ Not included

**Sample error cases:**
- REAL_0164: True=Java Developer, SVM=Python Developer, Majority=Java Developer
- REAL_0179: True=Java Developer, SVM=Web Designing, Majority=Java Developer
- REAL_0194: True=Java Developer, SVM=React Developer, Majority=Java Developer

**Error patterns:**
- Most SVM errors involve confusion between related developer roles
- Java Developer frequently confused with Python Developer, React Developer, Web Designing
- Majority baseline only correct on "Java Developer" samples (the most frequent class)

---

## 6. Changes Since Progress Report

### New Evaluation Components

1. **BERT Model Training:** ✅ COMPLETED
   - BERT was not trained at progress report time
   - Now fully trained with results in `runs/bert_finetuned/`
   - Validation: Accuracy=0.8229, Macro-F1=0.8881
   - Test: Accuracy=0.8800, Macro-F1=0.9238

2. **BERT Confusion Matrices:** ✅ NEW
   - `runs/bert_finetuned/confusion_val.png` - Present
   - `runs/bert_finetuned/confusion_test.png` - Present
   - Not present in progress report

3. **Error Analysis Scripts:** ✅ NEW
   - `src/eval/error_analysis.py` - Implemented
   - `runs/comparison/error_cases_val.csv` - Generated
   - `runs/comparison/error_cases_test.csv` - Generated
   - Not present in progress report

4. **Model Comparison Script:** ✅ NEW
   - `src/eval/compare_models.py` - Implemented
   - `runs/comparison/model_comparison.json` - Generated
   - Not present in progress report

5. **Top-k Metrics:** ✅ EXISTED (SVM only)
   - Top-1 and Top-3 accuracy computed for SVM
   - Not computed for BERT (no probability extraction implemented)
   - Present in progress report for SVM

### Metrics Extensions

- **Top-k accuracy:** Extended to include Top-3 (was only Top-1 in some contexts)
- **Error categorization:** New categories added (both_wrong, svm_wrong_bert_right, etc.) but BERT not yet included in error analysis
- **Per-class metrics:** Still not explicitly computed (only visible via confusion matrices)

### Visualization Updates

- **Confusion matrices:** Now generated for BERT (validation and test)
- **Error analysis:** New CSV outputs with detailed error cases
- **Comparison tables:** New formatted console output from `compare_models.py`

---

## 7. Visualization Inventory

### Confusion Matrices

1. **SVM Validation:** `runs/svm_tfidf/confusion_val.png` ✅ (569 KB)
2. **SVM Test:** `runs/svm_tfidf/confusion_test.png` ✅ (596 KB)
3. **BERT Validation:** `runs/bert_finetuned/confusion_val.png` ✅ (603 KB)
4. **BERT Test:** `runs/bert_finetuned/confusion_test.png` ✅ (600 KB)

**Total:** 4 confusion matrix visualizations

### Other Visualizations

- **Per-class F1 bar charts:** ❌ Not generated
- **Calibration curves:** ❌ Not generated
- **Top-k recall plots:** ❌ Not generated
- **Class distribution plots:** ❌ Not generated
- **Class size vs F1 plots:** ❌ Not generated
- **t-SNE/UMAP embeddings:** ❌ Not generated (transformer_embeddings.npy exists but unused)

---

## 8. Evaluation Artifacts Summary

### Metrics Files
- `runs/majority_baseline/metrics.json` ✅
- `runs/svm_tfidf/metrics.json` ✅
- `runs/bert_finetuned/metrics.json` ✅ (but comparison script looks for `bert_title/`)

### Comparison Files
- `runs/comparison/model_comparison.json` ✅ (but BERT shows null due to path mismatch)
- `runs/comparison/error_cases_val.csv` ✅
- `runs/comparison/error_cases_test.csv` ✅

### Visualization Files
- `runs/svm_tfidf/confusion_val.png` ✅
- `runs/svm_tfidf/confusion_test.png` ✅
- `runs/bert_finetuned/confusion_val.png` ✅
- `runs/bert_finetuned/confusion_test.png` ✅

### Missing Artifacts
- Per-class precision/recall/F1 JSON files: ❌
- Predictions parquet/CSV files: ❌
- Probability/confidence scores: ❌
- Calibration plots: ❌
- Top-k accuracy for BERT: ❌

---

## 9. Key Evaluation Findings

### Performance Comparison

| Model | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Test Top-3 |
|-------|--------------|--------------|---------------|---------------|-----------|
| **Majority** | 0.0571 | 0.0030 | 0.0571 | 0.0030 | N/A |
| **SVM** | 0.9000 | 0.9357 | 0.8857 | 0.9242 | 0.9743 |
| **BERT** | 0.8229 | 0.8881 | 0.8800 | 0.9238 | N/A |

### Observations

1. **SVM vs BERT:**
   - SVM has higher validation accuracy (0.900 vs 0.823)
   - SVM has higher test accuracy (0.886 vs 0.880)
   - BERT macro-F1 matches SVM macro-F1 on test (0.924 vs 0.924)
   - BERT shows better generalization (test > validation accuracy)

2. **Error rates:**
   - SVM validation: 10.0% error rate
   - SVM test: 11.4% error rate
   - BERT validation: ~17.7% error rate (estimated)
   - BERT test: 12.0% error rate

3. **Hard cases:**
   - Validation: 34 samples where both SVM and Majority are wrong (9.7%)
   - Test: 37 samples where both SVM and Majority are wrong (10.6%)
   - These represent genuinely difficult samples

4. **Class imbalance impact:**
   - 7 classes have < 5 examples in validation/test
   - Smallest classes: Technical Writer (2), Product Manager (2), Principal Engineer (2-3)
   - These likely have lower performance due to limited training data

---

## 10. Evaluation Code Locations

### Metric Computation
- `src/eval/metrics.py`: Core metric functions (accuracy, macro_f1, topk_from_scores, confusion)

### Model Evaluation
- `src/models/majority.py`: Majority baseline evaluation
- `src/models/train_svm.py`: SVM training and evaluation (includes `_evaluate_split`)
- `src/models/train_bert.py`: BERT training and evaluation (includes `compute_metrics`)

### Cross-Model Analysis
- `src/eval/compare_models.py`: Model comparison script
- `src/eval/error_analysis.py`: Error analysis and categorization script

---

## 11. Known Issues and Limitations

1. **BERT path mismatch:**
   - Comparison script expects `runs/bert_title/metrics.json`
   - Actual results in `runs/bert_finetuned/metrics.json`
   - Causes BERT metrics to show as null in comparison JSON

2. **BERT not included in error analysis:**
   - Error analysis CSVs do not include BERT predictions
   - Would need to re-run `error_analysis.py` with BERT model available

3. **Top-k not computed for BERT:**
   - BERT probabilities/logits not extracted for Top-k computation
   - Only SVM has Top-k metrics

4. **Per-class metrics not saved:**
   - No per-class precision/recall/F1 JSON files
   - Only visible via confusion matrix inspection

5. **No prediction files:**
   - Predictions not saved to parquet/CSV for later analysis
   - Only generated during evaluation and used for metrics

---

## 12. Evaluation Methodology Consistency

### Split Consistency
- ✅ All models use same splits from `data/processed/splits_v1.json`
- ✅ Same random seed (42) used throughout
- ✅ Stratified splits ensure class distribution maintained

### Label Encoding Consistency
- ✅ All models use same `LabelEncoder` from `models/label_encoder.joblib`
- ✅ Ensures consistent class mapping across models

### Metric Computation Consistency
- ✅ All models use same metric functions from `src.eval.metrics`
- ✅ Accuracy and macro-F1 computed identically for all models

### Evaluation Timing
- ✅ All models evaluated on same validation and test sets
- ✅ No data leakage (TF-IDF fit only on training data)

---

**END OF EVALUATION INVESTIGATION REPORT**

