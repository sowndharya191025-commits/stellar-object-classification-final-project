# Stellar Object Classification

Machine learning project for classifying celestial objects — **Star**, **Galaxy**, and **Quasar (QSO)** — using photometric and spectroscopic data from the Sloan Digital Sky Survey (SDSS).

## Overview

In the era of data-driven astronomy, the rapid growth of celestial survey data has created a need for intelligent, automated systems that can process vast amounts of information accurately and efficiently. This project builds and compares several machine learning models to classify stellar objects, aiming to support celestial cataloguing and large-scale astronomical data processing.

## Problem Statement

- Manual classification of massive astronomical catalogues is time-consuming and difficult to scale.
- Object classes can overlap in observed photometric properties.
- A reliable classifier must learn nonlinear relationships among sky position, magnitudes, and redshift.
- Goal: build and evaluate machine learning models that accurately identify each stellar-object class.

## Dataset

- **Source:** `star_classification.csv` from the Sloan Digital Sky Survey (SDSS)
- **Records:** 100,000
- **Columns:** 18 (15 feature columns + target)
- **Target classes:** GALAXY, STAR, QSO
- **Missing values:** 0

**Feature groups:**

| Group | Features |
|---|---|
| Positional | `alpha`, `delta` — celestial coordinates |
| Photometric | `u`, `g`, `r`, `i`, `z` — magnitudes across five wavelength bands |
| Spectroscopic | `redshift` — distance / cosmological motion indicator |
| Identification | `obj_ID`, `spec_obj_ID`, `run_ID`, `rerun_ID`, `cam_col`, `field_ID`, `plate`, `MJD`, `fiber_ID` |

**Class distribution:**

| Class | Count | Share |
|---|---|---|
| Galaxy | 59,445 | 59.4% |
| Star | 21,594 | 21.6% |
| QSO | 18,961 | 19.0% |

The dataset is moderately imbalanced, so a stratified train/test split was used to preserve class proportions.

## Exploratory Data Analysis

- Galaxy is the most dominant class, followed by Star, with QSO the least represented.
- `u`, `g`, and `z` photometric bands are skewed and compressed; `r` and `i` are closer to bell-shaped — feature scaling is needed due to differing ranges.
- **Redshift by class** is highly discriminative: Star ≈ 0, Galaxy ≈ 0.42, QSO ≈ 1.72 (broadest range).
- Photometric bands correlate strongly with one another.

## Data Preprocessing

- No missing values were found.
- Removed non-informative identification columns: `obj_ID`, `run_ID`, `rerun_ID`, `cam_col`, `field_ID`, `spec_obj_ID`, `plate`, `MJD`, `fiber_ID` — these don't aid class prediction, add noise, and risk overfitting.
- Target variable `class` (Star, Galaxy, Quasar) encoded numerically via label encoding.
- Features standardized using `StandardScaler`.
- Data split into 80% train / 20% test, with `stratify = y` and a fixed random state for reproducibility.

## Machine Learning Workflow

```
Raw data → Clean → Select features → Split → Train → Evaluate → Select
```

A reproducible pipeline ensures the same preprocessing logic is applied consistently during training and testing.

## Models Compared

| Model | Learning Type | Handles Non-Linearity | Interpretability |
|---|---|---|---|
| Random Forest | Ensemble (Tree-Based) | Yes | Moderate |
| XGBoost | Ensemble (Boosting) | Yes | Moderate |
| Gradient Boosting | Ensemble (Boosting) | Yes | Moderate |
| Decision Tree | Tree-Based | Yes | High |
| SVM | Kernel-Based | Yes | Low |
| KNN | Instance-Based | Yes | Low |
| AdaBoost | Ensemble (Boosting) | Yes | Moderate |

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Random Forest** | **0.98055** | **0.98046** | **0.98055** | **0.98044** |
| XGBoost | 0.97880 | 0.97871 | 0.97880 | 0.97870 |
| Gradient Boosting | 0.97765 | 0.97758 | 0.97765 | 0.97747 |
| Decision Tree | 0.96565 | 0.96582 | 0.96565 | 0.96573 |
| SVM | 0.96055 | 0.96084 | 0.96055 | 0.96034 |
| KNN | 0.92905 | 0.92987 | 0.92905 | 0.92867 |
| AdaBoost | 0.85910 | 0.88276 | 0.85910 | 0.86514 |

### Best Model: Random Forest — 98.05% Test Accuracy

- Highest test accuracy among all implemented models.
- Captures nonlinear relationships between photometric features and redshift.
- Robust to complex feature interactions; does not require feature scaling.
- Confusion matrix shows strong separation across all three classes:
  - Galaxy: 11,748 correct (117 → QSO, 24 → Star)
  - QSO: 3,548 correct (244 → Galaxy)
  - Star: 4,315 correct (4 → Galaxy)

### Top Important Features

`redshift` is by far the most important feature, followed by `z`, `g`, `i`, and `u`. `r`, `plate`, `MJD`, `delta`, and `alpha` contribute comparatively little.

## Key Findings

- The dataset is large and information-rich, supporting robust supervised learning.
- Class imbalance makes stratified splitting and weighted evaluation important.
- Photometric measurements provide complementary spectral information.
- Redshift is a highly informative discriminator, especially for separating stars from distant objects.
- Random Forest delivered the strongest overall predictive performance in this experiment.

## Future Scope

- Hyperparameter optimization and cross-validation.
- Feature engineering using colour indices such as u−g and g−r.
- Exploring gradient boosting variants, neural networks, and explainable AI.
- Deployment for real-time catalogue screening.

## Project Structure

```
├── data/        # Dataset files
├── model/       # Trained model artifacts
├── notebook/    # EDA, preprocessing, and modelling notebooks
├── reports/     # Visualizations, dashboard, and project report
├── src/         # Source code / reusable scripts
```

## How to Run

```bash
git clone https://github.com/sowndharya191025-commits/Stellar-object-classification.git
cd Stellar-object-classification
pip install -r requirements.txt
```

Open the notebook in the `notebook/` folder and run the cells in order.

## Requirements

- Python 3.x
- pandas, numpy
- scikit-learn
- xgboost
- matplotlib, seaborn

## Author

**Sowndharya**
