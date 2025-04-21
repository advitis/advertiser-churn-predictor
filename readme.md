# Advertiser Churn Predictor

A modular pipeline to generate synthetic advertiser data, train an XGBoost churn‑prediction model, evaluate performance, and explain individual predictions with SHAP.

## Project Structure

- `data/`
  – Contains the synthetic dataset CSV
- `preprocessing.py`
  – `load_data()`: loads, splits, and scales features
- `training.py`
  – `train_model()`: fits XGBoost with optional early stopping
- `evaluation.py`
  – `evaluate()`: prints AUC‑ROC/AUC‑PR and risk‑band precision/recall
- `scoring.py`
  – `explain_and_score()`: assigns churn‑prob, risk category, and main SHAP driver
- `run_pipeline.py`
  – `main()`: orchestrates the full end‑to‑end workflow

## Setup

1. **Clone** this repo and `cd` into it.
2. **Install** dependencies:
   ```bash
   pip install -r requirements.txt
