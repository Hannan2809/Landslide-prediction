from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, classification_report
import joblib
from xgboost import XGBClassifier

"""Standalone XGBoost experiment for GriffonPeak01. Includes its own preprocessing and evaluation."""

# =========================
# 1. Dataset and constants
# =========================
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "GriffonPeak01.csv"
TARGET = "Movements_t+1"
BASE_SENSORS = [
    "Tem", "Hum", "Pressure", "Rain", "Light", "Ax", "Ay", "Az",
    "Wx", "Wy", "Wz", "Moisture"
]
EXPECTED_FEATURE_COUNT = 104
GAP_MINUTES = 30
TEST_FRACTION = 0.10
RANDOM_STATE = 42

# =========================
# 2. Load and preprocess
# =========================
def load_and_engineer_features(csv_path: Path, gap_minutes: int = GAP_MINUTES):
    """Standalone feature engineering; no shared project pipeline is imported."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = ["Date", TARGET, *BASE_SENSORS]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=["Date", TARGET]).sort_values("Date", kind="stable").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid labelled rows after Date/target cleaning.")
    if not np.isclose(df[TARGET], np.round(df[TARGET])).all():
        raise ValueError("Target labels must be integer-valued.")
    df[TARGET] = df[TARGET].astype(int)

    X = df[BASE_SENSORS].apply(pd.to_numeric, errors="coerce").copy()
    dates = df["Date"]
    minute_of_day = dates.dt.hour * 60 + dates.dt.minute
    day_of_year = dates.dt.dayofyear
    weekday = dates.dt.dayofweek

    X["time_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    X["time_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    X["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    X["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    X["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)

    gaps = dates.diff().dt.total_seconds().div(60)
    segment_id = gaps.gt(gap_minutes).fillna(True).cumsum()

    X["accel_magnitude"] = np.sqrt(X["Ax"]**2 + X["Ay"]**2 + X["Az"]**2)
    X["angular_rate_magnitude"] = np.sqrt(X["Wx"]**2 + X["Wy"]**2 + X["Wz"]**2)

    for sensor in BASE_SENSORS:
        grouped = X[sensor].groupby(segment_id, sort=False)
        for lag in (1, 2, 3):
            X[f"{sensor}_lag{lag}"] = grouped.shift(lag)
        X[f"{sensor}_rollmean3"] = grouped.transform(lambda s: s.rolling(3, min_periods=1).mean())
        X[f"{sensor}_rollstd3"] = grouped.transform(lambda s: s.rolling(3, min_periods=2).std())
        X[f"{sensor}_rollmean6"] = grouped.transform(lambda s: s.rolling(6, min_periods=1).mean())
        X[f"{sensor}_diff1"] = grouped.diff(1)

    X = X.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    if X.shape[1] != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(f"Feature count mismatch: expected {EXPECTED_FEATURE_COUNT}, got {X.shape[1]}")
    y = df[TARGET].reset_index(drop=True)
    return X, y, dates.reset_index(drop=True)

# =========================
# 3. Chronological split
# =========================
def chronological_train_test_split(X, y, dates, test_fraction=TEST_FRACTION):
    split_index = int(len(X) * (1.0 - test_fraction))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    date_train, date_test = dates.iloc[:split_index], dates.iloc[split_index:]
    return X_train, X_test, y_train, y_test, date_train, date_test

# =========================
# 4. Train-only imputation
# =========================
def preprocess_train_test(X_train, X_test, scale: bool):
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train_proc = scaler.fit_transform(X_train_imp)
        X_test_proc = scaler.transform(X_test_imp)
    else:
        X_train_proc, X_test_proc = X_train_imp, X_test_imp
    return X_train_proc, X_test_proc, imputer, scaler

# =========================
# 5. Model definition
# =========================
def build_model():
    return XGBClassifier(
        objective="multi:softprob", num_class=3, n_estimators=400,
        max_depth=4, learning_rate=0.04, subsample=0.85,
        colsample_bytree=0.85, min_child_weight=1, reg_lambda=2.0,
        gamma=0.1, eval_metric="mlogloss", random_state=42, n_jobs=-1
    )

# =========================
# 6. Evaluation and artifacts
# =========================
def evaluate_and_save(model, X_test, y_test, dates_test, feature_names, output_dir: Path, train_rows: int):
    predictions = np.asarray(model.predict(X_test)).astype(int).reshape(-1)
    labels = sorted(np.unique(np.concatenate([np.asarray(y_test), predictions])).tolist())
    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, labels=labels, average="macro", zero_division=0)
    precision, recall, class_f1, support = precision_recall_fscore_support(y_test, predictions, labels=labels, zero_division=0)
    cm = confusion_matrix(y_test, predictions, labels=labels)
    summary = {
        "model": "XGBoost", "n_features": len(feature_names), "train_rows": int(train_rows),
        "test_rows": int(len(y_test)), "test_start": str(dates_test.iloc[0]), "test_end": str(dates_test.iloc[-1]),
        "accuracy": float(accuracy), "balanced_accuracy": float(balanced_accuracy), "macro_f1": float(macro_f1)
    }
    pd.DataFrame([summary]).to_csv(output_dir / "overall_metrics.csv", index=False)
    pd.DataFrame({"class": labels, "precision": precision, "recall": recall, "f1": class_f1, "support": support}).to_csv(output_dir / "class_metrics.csv", index=False)
    pd.DataFrame(cm, index=[f"true_{c}" for c in labels], columns=[f"pred_{c}" for c in labels]).to_csv(output_dir / "confusion_matrix.csv")
    pd.DataFrame({"Date": dates_test.astype(str).values, "y_true": np.asarray(y_test), "y_pred": predictions}).to_csv(output_dir / "test_predictions.csv", index=False)
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, predictions, labels=labels, zero_division=0))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("XGBoost — Confusion Matrix")
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(output_dir / "confusion_matrix.png", dpi=200); plt.close(fig)
    print("\\nFINAL HOLDOUT RESULTS — XGBoost")
    print(f"Accuracy:          {accuracy:.6f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.6f}")
    print(f"Macro-F1:         {macro_f1:.6f}")
    print(pd.DataFrame({"class": labels, "precision": precision, "recall": recall, "f1": class_f1, "support": support}).to_string(index=False))
    return summary

def main():
    parser = argparse.ArgumentParser(description="Standalone XGBoost final chronological holdout experiment.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to GriffonPeak01 CSV")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "XGBoost", help="Output folder")
    parser.add_argument("--gap-minutes", type=float, default=GAP_MINUTES, help="Gap threshold for segment reset")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    X, y, dates = load_and_engineer_features(args.data, args.gap_minutes)
    X_train, X_test, y_train, y_test, date_train, date_test = chronological_train_test_split(X, y, dates)
    X_train_proc, X_test_proc, imputer, scaler = preprocess_train_test(X_train, X_test, scale=False)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model = build_model()
    model.fit(X_train_proc, y_train, sample_weight=sample_weight)

    joblib.dump({"model": model, "imputer": imputer, "scaler": scaler, "feature_names": list(X.columns)}, args.output / "trained_pipeline.joblib")
    evaluate_and_save(model, X_test_proc, y_test, date_test, list(X.columns), args.output, len(X_train))
    metadata = {
        "model": "XGBoost", "feature_names": list(X.columns), "feature_count": int(X.shape[1]),
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "split": "chronological_first_90_percent_train_last_10_percent_test",
        "gap_minutes": args.gap_minutes, "scale_features": False,
        "imbalance_handling": 'balanced sample weights',
        "parameters": str(model.get_params())
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if hasattr(model, "feature_importances_"):
        pd.DataFrame({"feature": list(X.columns), "importance": model.feature_importances_}).sort_values("importance", ascending=False).to_csv(args.output / "feature_importance.csv", index=False)
    print(f"\\nArtifacts saved to: {args.output.resolve()}")

if __name__ == "__main__":
    main()
