# PhishGuard - Email Phishing Detection (ML + Flask UI)

A production-style machine learning system for detecting phishing emails using a Logistic Regression classifier over word-count features, with an explainable Flask web UI. The project implements a complete pipeline from dataset preparation to model training, serialization, and an interactive web interface with token-level contribution explanations.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Objectives](#objectives)
3. [System Architecture](#system-architecture)
4. [Dataset](#dataset)
5. [Machine Learning Pipeline](#machine-learning-pipeline)
   - [Data Preprocessing](#data-preprocessing)
   - [Feature Handling](#feature-handling)
   - [Model Selection and Training](#model-selection-and-training)
   - [Evaluation Metrics](#evaluation-metrics)
6. [Explainability](#explainability)
7. [Web UI (Flask)](#web-ui-flask)
8. [Installation and Usage](#installation-and-usage)
9. [Results and Performance](#results-and-performance)
10. [Technical Specifications](#technical-specifications)
11. [Project Structure](#project-structure)
12. [Challenges and Solutions](#challenges-and-solutions)
13. [Methodology](#methodology)

---

## Project Overview

This project detects whether an input email is phishing or legitimate. It leverages a traditional NLP approach using precomputed word-count features and a linear classifier. The system is designed for clarity, reproducibility, strong performance, and explainability suitable for academic reports.

**Technology Stack:**

- ML Pipeline: Python 3.14, scikit-learn, pandas, numpy
- Model: Logistic Regression (balanced class weights)
- Serving: Flask web app with custom CSS
- Serialization: joblib

---

## Objectives

### Primary Objective

Build a binary classifier that predicts whether an email is phishing (1) or legitimate (0) with high test accuracy and recall.

### Secondary Objectives

1. Provide an explainable UI highlighting tokens that drive the prediction.
2. Maintain a clean, reproducible training pipeline with frozen vocabulary order.
3. Offer an easy-to-deploy Flask app for local use or Render.

### Success Criteria

- Test accuracy ≥ 0.96
- High phishing recall (≥ 0.95) to minimize missed attacks
- Simple, defensible model with clear justifications

---

## System Architecture

```
CSV (word counts)
    → Phase 1: Data prep (split, vocab)
    → Phase 2: Baselines (CV)
    → Phase 3: Final training + serialization
    → Flask UI (load vocab + model)
    → Inference (text → count vector → prediction + explanation)
```

---

## Dataset

- Location: `data/emails.csv`
- Shape summary (from prep):
  - Rows: 5,172
  - Features: 3,000 (word tokens)
  - Label column: `Prediction` (0 = legitimate, 1 = phishing)
  - Dropped columns: `Email No.`
  - Class distribution: 3,672 legitimate, 1,500 phishing

The header contains a fixed vocabulary of tokens; each email row provides integer word counts for those tokens.

---

## Machine Learning Pipeline

### Data Preprocessing

- Script: `src/data_prep.py`
- Actions:
  - Load `data/emails.csv`
  - Drop `Email No.`
  - Separate `X` (all token columns) and `y = Prediction`
  - Validate binary labels {0,1}
  - Stratified train/test split (80/20)
  - Save artifacts to `data/processed/`:
    - `train.csv`, `test.csv`
    - `vocab.json` (ordered feature names)
    - `summary.json` (dataset summary)

### Feature Handling

- Feature space = counts for 3,000 tokens (from CSV header)
- Inference uses the exact same token order loaded from `vocab.json`
- No TF–IDF or additional scaling (Logistic Regression handles counts well; class_weight balances classes)

### Model Selection and Training

- Baselines (CV): `src/train_baselines.py`
  - Models: LogisticRegression (balanced), LinearSVC (calibrated), MultinomialNB
  - 5-fold Stratified CV on training split
  - Selected by mean accuracy → LogisticRegression
- Final training: `src/train_final.py`
  - Train LogisticRegression(max_iter=2000, solver=liblinear, class_weight=balanced)
  - Evaluate on held-out test set
  - Serialize model to `models/model.joblib`; save `models/metadata.json`

### Evaluation Metrics

From the latest run:

- Train metrics: accuracy 0.999, precision 0.998, recall 1.000, F1 0.999
- Test metrics: accuracy 0.981, precision 0.952, recall 0.983, F1 0.967

Interpretation:

- High recall minimizes missed phishing emails (false negatives)
- High precision limits false alarms on legitimate emails
- A small train→test gap indicates good generalization

---

## Explainability

- Method: Linear model coefficients × token counts
- For each input, the UI lists the top tokens pushing the decision toward phishing (positive contributions) and toward legitimate (negative contributions)
- Implementation: `src/inference.py`
  - `text_to_vector(text, vocab)`: lowercase, split on non-alphanumeric, count tokens, map to vocab order
  - `predict_text(text, vocab, model)`: returns label, probability, and an explanation object with token contributions and derived probability from the full logit

---

## Web UI (Flask)

- App entry: `src/app.py`
- Templates: `templates/base.html`, `templates/index.html`
- Static assets: `static/app.css`, `static/app.js`

Features:

- Two-column layout (left: input form; right: results)
- Centered heading; responsive design
- Result card showing label badge (phishing/legitimate) and confidence
- Token-level explanation tables (token, count, contribution)
- UX niceties: loading state on submit, clear textarea, copy results

---

## Installation and Usage

### Prerequisites

- Python 3.14+
- Recommended: virtual environment (venv)

### Install dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Ensure the dataset exists at `data/emails.csv`.

### Phase 1: Data prep

```bash
python src/data_prep.py --input data/emails.csv --output_dir data/processed --test_size 0.2 --random_state 42
```

### Phase 2: Baselines (optional but recommended)

```bash
python src/train_baselines.py --train_csv data/processed/train.csv --report_out reports/baselines.json --cv_splits 5 --random_state 42
```

### Phase 3: Train final model

```bash
python src/train_final.py \
  --train_csv data/processed/train.csv \
  --test_csv data/processed/test.csv \
  --vocab_json data/processed/vocab.json \
  --models_dir models \
  --random_state 42
```

### Run the web app

```bash
python -m src.app
```

Open http://localhost:5000, paste email text, and click Analyze.

---

## Results and Performance

Summary of current best model (Logistic Regression):

```
Test Accuracy   0.981
Test Precision  0.952
Test Recall     0.983
Test F1         0.967
```

These metrics reflect excellent performance for a word-count model, especially on recall.

---

## Technical Specifications

### Dataset

- Samples: 5,172 emails (3,672 legit, 1,500 phishing)
- Features: 3,000 tokens (word counts)
- Target: `Prediction` (0/1)

### Model

- Algorithm: LogisticRegression
- Solver: liblinear
- Class weights: balanced
- Max iterations: 2000
- Serialization: joblib

### Serving

- Framework: Flask
- Explainability: coefficient × count per token
- No external fonts or CSS frameworks; custom CSS only

---

## Project Structure

```
amul-mle/
├── data/
│   ├── emails.csv                 # Provided dataset (word counts)
│   └── processed/                 # Generated: train/test, vocab, summary
├── models/                        # Generated: model.joblib, metadata.json
├── reports/                       # Generated: baselines.json (optional)
├── src/
│   ├── __init__.py                # Package marker
│   ├── app.py                     # Flask app
│   ├── data_prep.py               # Phase 1: prep & artifacts
│   ├── inference.py               # Text→vector + prediction + explanation
│   ├── train_baselines.py         # Phase 2: CV baselines
│   └── train_final.py             # Phase 3: final training & eval
├── templates/
│   ├── base.html                  # Base layout
│   └── index.html                 # Main page with form & results
├── static/
│   ├── app.css                    # Complete custom theme
│   └── app.js                     # Loading state, clear, copy
├── requirements.txt               # Python dependencies
├── .gitignore                     # Ignore data/artifacts/venv/caches
└── README.md                      # This file
```

---

## Challenges and Solutions

1. Class imbalance (legit > phishing)
   - Solution: `class_weight='balanced'` in Logistic Regression; stratified split; chosen metrics emphasize recall.
2. Feature alignment at inference
   - Solution: Persist `vocab.json` (ordered features) and strict vectorization against this order.
3. Explainability for academic reporting
   - Solution: Linear model with coefficient×count contributions; top positive/negative tokens surfaced in the UI.
4. Simple, reliable deployment
   - Solution: Minimal dependencies, joblib serialization, Flask UI without external fonts/frameworks.

---

## Methodology

### Strengths

1. Clear, modular pipeline (prep → CV → final → serve)
2. Strong generalization on held-out set with high recall
3. Explainable predictions at token level
4. Deterministic and reproducible (fixed seeds, persisted vocab)

### Limitations & Future Work

- Current tokenizer is simple (whitespace/punctuation split). Future: URL normalization, header parsing, lemmatization.
- Features are bag-of-words counts. Future: add character n-grams, TF–IDF, or modern embeddings for robustness.
- Threshold is 0.5. Future: calibrate threshold to optimize for recall/precision trade-offs per use case.

---

Happy detecting! If you deploy to Render, ensure `models/` and `data/processed/vocab.json` are available at build/runtime, or add a setup step to train and persist artifacts.


