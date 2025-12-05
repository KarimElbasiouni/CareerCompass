# BERT Training Setup for Google Colab

This guide provides step-by-step commands to train the BERT model on Google Colab.

## Step 1: Clone the Repository

```bash
!git clone https://github.com/KarimElbasiouni/CareerCompass.git
!cd CareerCompass
```

## Step 2: Install Dependencies

```bash
!cd CareerCompass && pip install -r requirements.txt
```

This will install:
- torch
- transformers
- pandas, numpy, scikit-learn
- matplotlib, joblib, pyarrow

## Step 3: Train BERT Model

```bash
!cd CareerCompass && python -m src.models.train_bert
```

**Optional arguments:**
- `--epochs 3` (default: 3)
- `--batch-size 8` (default: 8)
- `--eval-batch-size 16` (default: 16)
- `--learning-rate 2e-5` (default: 2e-5)

**Example with custom settings:**
```bash
!cd CareerCompass && python -m src.models.train_bert --epochs 5 --batch-size 16 --learning-rate 3e-5
```

**What happens:**
1. Loads data from `data/processed/resumes_v1.parquet`
2. Loads train/val/test splits from `data/processed/splits_v1.json`
3. Downloads BERT model from Hugging Face (bert-base-uncased)
4. Trains the model for specified epochs
5. Evaluates on validation and test sets
6. Saves model to `models/bert_title/`
7. Saves metrics and confusion matrices to `runs/bert_title/`

## Step 4: Run Comparison and Error Analysis (Optional on Colab)

If you want to run these on Colab:

```bash
!cd CareerCompass && python -m src.eval.compare_models
!cd CareerCompass && python -m src.eval.error_analysis --split val
!cd CareerCompass && python -m src.eval.error_analysis --split test
```

## After Training Locally

After training completes on Colab, you'll have these folders/files:

### Generated Artifacts:

1. **`models/bert_title/`** - Contains:
   - `pytorch_model.bin` or `model.safetensors` (model weights)
   - `config.json` (model configuration)
   - `tokenizer_config.json` and `vocab.txt` (tokenizer files)

2. **`runs/bert_title/`** - Contains:
   - `metrics.json` (validation and test metrics)
   - `confusion_val.png` (validation confusion matrix)
   - `confusion_test.png` (test confusion matrix)
   - Training checkpoints (if any)

### Download and Transfer to Local Repo:

1. **Download from Colab:**
   - Right-click `models/bert_title/` folder → Download
   - Right-click `runs/bert_title/` folder → Download

2. **Copy to your local repo:**
   - Extract the downloaded folders
   - Copy `bert_title/` into your local `CareerCompass/models/` directory
   - Copy `bert_title/` into your local `CareerCompass/runs/` directory

3. **Run evaluation locally:**
   ```bash
   python -m src.eval.compare_models
   python -m src.eval.error_analysis --split val
   python -m src.eval.error_analysis --split test
   ```

These commands will now include BERT in the comparison and error analysis!

