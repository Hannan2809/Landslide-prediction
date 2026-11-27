# GriffonPeak01 — Seven Standalone Model Experiments

This repository contains **seven independent scripts**. Every script contains its own:
- data loading and validation
- 12-to-104 feature engineering
- 30-minute gap segmentation
- chronological 90/10 split
- train-only median imputation
- model-specific preprocessing
- model training
- final holdout evaluation
- result/artifact saving

**No shared/common preprocessing pipeline is imported.** Each model script is independently readable and runnable.

## Models

1. LightGBM — `models/lightgbm_final.py`
2. Random Forest — `models/random_forest_final.py`
3. XGBoost — `models/xgboost_final.py`
4. CatBoost — `models/catboost_final.py`
5. Extra Trees — `models/extra_trees_final.py`
6. Linear SVM — `models/linear_svm_final.py`
7. RBF SVM — `models/rbf_svm_final.py`

## Dataset setup

Place the CSV at:

```text
data/GriffonPeak01.csv
```

The scripts expect the timestamp, target and 12 sensor columns:
- `Date`
- `Movements_t+1`
- `Tem`, `Hum`, `Pressure`, `Rain`, `Light`, `Moisture`
- `Ax`, `Ay`, `Az`, `Wx`, `Wy`, `Wz`

Do not commit the dataset unless you have permission to distribute it.

## Installation

```bash
python -m venv .venv
```

Activate the environment, then:

```bash
pip install -r requirements.txt
```

## Run one model

From the repository root:

```bash
python models/lightgbm_final.py
python models/random_forest_final.py
python models/xgboost_final.py
python models/catboost_final.py
python models/extra_trees_final.py
python models/linear_svm_final.py
python models/rbf_svm_final.py
```

Each script supports custom data and output paths:

```bash
python models/lightgbm_final.py --data path/to/GriffonPeak01.csv --output results/my_lightgbm
```

## Feature engineering: 104 columns

| Group | Count |
|---|---:|
| Raw sensors retained | 12 |
| Cyclical time features | 6 |
| IMU magnitudes | 2 |
| Lags 1, 2, 3 for each sensor | 36 |
| Rolling mean window 3 | 12 |
| Rolling standard deviation window 3 | 12 |
| Rolling mean window 6 | 12 |
| First differences | 12 |
| **Total** | **104** |

Temporal features are observation-based (row-based), not fixed-duration. Lags and rolling history reset after gaps greater than 30 minutes.

## Final evaluation protocol

- Sort rows chronologically.
- Train on the first 90% and test on the final 10%.
- Fit median imputation on the training set only.
- Standardize features only for Linear SVM and RBF SVM; scaler is fit on training data only.
- Tree models use balanced sample weights; SVM models use `class_weight='balanced'`.
- Report Accuracy, Balanced Accuracy, Macro-F1, class-wise precision/recall/F1, confusion matrix and predictions.
- Tree models additionally save feature importance when the estimator provides it.

**Caution:** The last 10% test partition has been used in previous experiments and should not be treated as a pristine untouched test set. Class 1 and Class 2 support are small, so class-wise metrics are uncertain.

## Output structure

Each model writes to a dedicated folder under `results/<model_name>/`:
- `overall_metrics.csv`
- `class_metrics.csv`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `test_predictions.csv`
- `classification_report.txt`
- `run_metadata.json`
- `feature_importance.csv` (tree models, when available)
- `trained_pipeline.joblib`

## Hyperparameters

The scripts use the selected fixed configurations documented in the model code. They are not performing hyperparameter search at runtime. If package versions differ, results may not exactly reproduce prior scores.
