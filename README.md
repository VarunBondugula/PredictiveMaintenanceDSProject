# Predictive Maintenance (AI4I 2020) — Supervised Failure Prediction

A supervised machine-learning project that predicts machine failure from industrial sensor readings to support predictive maintenance.

## Overview
Manufacturing equipment failures can cause production stoppages, expensive repairs, and downstream delivery delays. This project trains and evaluates supervised classification models on historical milling-machine sensor data to detect failure-prone cycles early, enabling maintenance teams to intervene before breakdowns occur.

## Dataset
- **AI4I 2020 Predictive Maintenance** dataset
- Inputs include operational measurements such as:
  - Air temperature (K)
  - Process temperature (K)
  - Rotational speed (rpm)
  - Torque (Nm)
  - Tool wear (min)
  - Product type (categorical: L / M / H)
- Target:
  - **Machine failure** (1=failure, 0=normal)

## Method (CRISP-DM)
1. **Data cleaning & EDA**
   - Load CSV, validate schema, check missing values/outliers
   - Explore feature distributions and failure frequency (class imbalance)
2. **Feature engineering**
   - Temperature redundancy analysis (multicollinearity)
   - Try two alternatives to the raw temperature pair:
     - **PCA temperature component**
     - **Temperature difference** = process temp − air temp (interpretable)
3. **Modeling (Supervised Classification)**
   - Baseline: Decision Tree (interpretability)
   - Primary model: **Random Forest**
   - Also explored: Logistic Regression / XGBoost (as extensions)
4. **Imbalance handling**
   - Failures are rare (~3–4%), so we apply **SMOTE** on the training set to improve minority-class learning.
5. **Evaluation**
   - Hold-out split: **80% train / 20% test (stratified)**
   - Metrics: Accuracy, Precision, Recall, F1 (failure class), ROC AUC
   - Stability check: **5-fold CV AUC**

## Key Results (Random Forest Variants)
| Variant | ROC AUC | Precision (fail) | Recall (fail) | F1 (fail) | Notes |
|---|---:|---:|---:|---:|---|
| Baseline (raw temps) | 0.9637 | 0.43 | 0.78 | 0.56 | Strong separability, precision limited by rarity |
| PCA temp component | 0.9466 | 0.35 | 0.76 | — | Slight drop; PCA likely removed some useful signal |
| **Temp difference feature** | **0.9633** | **0.45** | **0.84** | **0.58** | Best overall balance; interpretable feature |

**Cross-validation (AUC):**
- Baseline: **0.9971 ± 0.0007**
- PCA: **0.9936 ± 0.0004**
- Temp difference: **0.9967 ± 0.0007**

**Multicollinearity note (VIF):**
Air/process temperature showed extremely high VIF values, motivating the PCA and temperature-difference experiments.

## Contributors
- Danesh Pritesh Patel
- Nideesh Madda
- Varun Bondugula
