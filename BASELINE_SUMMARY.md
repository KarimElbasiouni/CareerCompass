# CareerCompass: Baseline & Error Analysis Summary

## 1. Metrics Table (Majority vs SVM vs BERT)

| Metric | Majority | SVM | BERT |
|--------|----------|-----|------|
| **Val Accuracy** | 0.0571 | 0.9000 | N/A |
| **Val Macro-F1** | 0.0030 | 0.9357 | N/A |
| **Test Accuracy** | 0.0571 | 0.8857 | N/A |
| **Test Macro-F1** | 0.0030 | 0.9242 | N/A |
| **Test Top-1** | N/A | 0.8857 | N/A |
| **Test Top-3** | N/A | 0.9743 | N/A |

**Notes:**
- SVM hyperparameter: C = 2.0 (best from grid search)
- BERT metrics are N/A because BERT hasn't been trained in this environment

## 2. Available Visualizations

| Visualization | Status | Path |
|---------------|--------|------|
| SVM Validation Confusion Matrix | **PRESENT** | `runs/svm_tfidf/confusion_val.png` |
| SVM Test Confusion Matrix | **PRESENT** | `runs/svm_tfidf/confusion_test.png` |
| BERT Validation Confusion Matrix | **MISSING** | `runs/bert_title/confusion_val.png` |
| BERT Test Confusion Matrix | **MISSING** | `runs/bert_title/confusion_test.png` |

No comparison plots found in `runs/comparison/`.

## 3. Error Cases — Validation Split

**File:** `runs/comparison/error_cases_val.csv` ✅ **PRESENT**

**Summary:**
- **Total samples:** 350
- **SVM correct:** 315 (90.0%)
- **SVM wrong:** 35 (10.0%)
- **Majority correct:** 20 (5.7%)
- **Majority wrong:** 330 (94.3%)
- **Hard cases (both wrong):** 34

**Sample Error Cases:**

| resume_id | true_label | svm_pred | majority_pred | svm_correct | majority_correct |
|-----------|------------|----------|---------------|-------------|------------------|
| REAL_0191 | Java Developer | React Developer | Java Developer | False | True |
| REAL_0203 | Python Developer | DevOps | Java Developer | False | False |
| REAL_0232 | Python Developer | Java Developer | Java Developer | False | False |

**Observations:**
- Most SVM errors involve confusion between related developer roles (Java, Python, React, DevOps)
- Majority baseline only gets 20 samples correct (all are "Java Developer", the most frequent class)
- 34 samples are misclassified by both models (hard cases)

## 4. Error Cases — Test Split

**File:** `runs/comparison/error_cases_test.csv` ✅ **PRESENT**

**Summary:**
- **Total samples:** 350
- **SVM correct:** 310 (88.6%)
- **SVM wrong:** 40 (11.4%)
- **Majority correct:** 20 (5.7%)
- **Majority wrong:** 330 (94.3%)
- **Hard cases (both wrong):** 37

**Sample Error Cases:**

| resume_id | true_label | svm_pred | majority_pred | svm_correct | majority_correct |
|-----------|------------|----------|---------------|-------------|------------------|
| REAL_0164 | Java Developer | Python Developer | Java Developer | False | True |
| REAL_0179 | Java Developer | Web Designing | Java Developer | False | True |
| REAL_0194 | Java Developer | React Developer | Java Developer | False | True |

**Observations:**
- Test set shows slightly more errors than validation (40 vs 35), indicating minor overfitting
- Similar error patterns: confusion between related developer roles
- 37 hard cases where both models fail (slightly more than validation's 34)

## 5. Short Takeaways

- **SVM significantly outperforms Majority baseline:**
  - Test accuracy: 88.6% vs 5.7% (15.5x improvement)
  - Test macro-F1: 0.924 vs 0.003 (308x improvement)
  - Top-3 accuracy of 97.4% shows SVM is very confident in its top predictions

- **Error distribution:**
  - SVM errors are relatively sparse (10-11% of samples)
  - Most errors involve semantically similar job titles (Java vs Python vs React Developer)
  - 34-37 hard cases (9.7-10.6% of samples) where both models fail, suggesting these may be genuinely ambiguous resumes

- **BERT status:**
  - BERT hasn't been trained in this environment, so comparison is currently Majority vs SVM only
  - When BERT is trained, the error analysis script will automatically include it and show:
    - Cases where BERT improves over SVM (SVM wrong / BERT right)
    - Cases where BERT regresses (SVM right / BERT wrong)

- **Model performance:**
  - SVM shows strong generalization with close validation/test performance (90.0% vs 88.6% accuracy)
  - The 1.4 percentage point drop from validation to test is minimal, indicating good generalization
  - Top-3 accuracy of 97.4% suggests that even when the top prediction is wrong, the correct label is usually in the top 3 predictions

