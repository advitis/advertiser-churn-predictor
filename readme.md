# Advertiser Churn Predictor

This project simulates a churn prediction pipeline for advertisers in a SaaS-like ecosystem, inspired by Pinterest’s ML approach to proactive advertiser retention. I was curious to test similar concepts hands-on.

I engineered behavioral features to capture engagement decay, trained classifiers under class imbalance, and prioritized interpretability using SHAP. Beyond performance metrics, I explored how churn probability can guide retention strategy, aligning machine learning outputs with business impact.

This project is part of an ongoing effort to bridge industry-relevant data science with real-world media and marketing applications.

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
