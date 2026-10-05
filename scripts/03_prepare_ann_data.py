from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "Data_Preparation" / "raw" / "Dataset_ATS_v2.csv"
OUT_DIR = BASE_DIR / "Predictive_Modeling"
DATA_DIR = OUT_DIR / "data"
PREPROCESSOR_DIR = OUT_DIR / "preprocessing"
DOC_DIR = OUT_DIR / "docs"

TARGET_COLUMN = "Churn"
EXPECTED_TARGET_VALUES = {"No", "Yes"}
RANDOM_STATE = 42
TEST_SIZE = 0.20


def add_check(checks: list[dict[str, str]], name: str, status: str, evidence: str) -> None:
    checks.append({"Check": name, "Status": status, "Evidence": evidence})


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSOR_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    checks: list[dict[str, str]] = []

    missing_values = int(df.isna().sum().sum())
    exact_duplicate_rows = int(df.duplicated().sum())
    duplicate_rate = exact_duplicate_rows / len(df)
    target_values = set(df[TARGET_COLUMN].dropna().unique())

    add_check(
        checks,
        "Dataset loaded",
        "PASS",
        f"{len(df):,} rows and {df.shape[1]} columns loaded from the official dataset.",
    )
    add_check(
        checks,
        "Missing values",
        "PASS" if missing_values == 0 else "FAIL",
        f"{missing_values} missing values found.",
    )
    add_check(
        checks,
        "Target values",
        "PASS" if target_values == EXPECTED_TARGET_VALUES else "FAIL",
        f"Observed target values: {sorted(target_values)}.",
    )
    add_check(
        checks,
        "Identical rows",
        "REVIEW" if exact_duplicate_rows else "PASS",
        (
            f"{exact_duplicate_rows} identical rows ({duplicate_rate:.2%}) were retained. "
            "The source has no customer identifier, so identical profiles cannot be proven to be accidental duplicates."
        ),
    )

    y = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    x = df.drop(columns=[TARGET_COLUMN])

    categorical_columns = x.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_columns = x.select_dtypes(include=["number"]).columns.tolist()

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    x_train_array = preprocessor.fit_transform(x_train_raw)
    x_test_array = preprocessor.transform(x_test_raw)
    feature_names = preprocessor.get_feature_names_out().tolist()

    x_train = pd.DataFrame(x_train_array, columns=feature_names, index=x_train_raw.index)
    x_test = pd.DataFrame(x_test_array, columns=feature_names, index=x_test_raw.index)

    train_set = x_train.copy()
    train_set[TARGET_COLUMN] = y_train
    test_set = x_test.copy()
    test_set[TARGET_COLUMN] = y_test

    add_check(
        checks,
        "Split integrity",
        "PASS" if set(x_train.index).isdisjoint(x_test.index) else "FAIL",
        f"Training rows: {len(train_set):,}; testing rows: {len(test_set):,}; stratified 80/20 split.",
    )
    add_check(
        checks,
        "Target leakage",
        "PASS" if TARGET_COLUMN not in feature_names else "FAIL",
        "Churn is excluded from all model input features.",
    )
    add_check(
        checks,
        "Identifier leakage",
        "PASS" if "CustomerID" not in feature_names else "FAIL",
        "No generated or source customer identifier is used as a model feature.",
    )
    add_check(
        checks,
        "Preprocessing fit",
        "PASS",
        "Encoder and scaler are fitted on training data only; the fitted pipeline is then applied to testing data.",
    )
    add_check(
        checks,
        "Feature alignment",
        "PASS" if list(x_train.columns) == list(x_test.columns) else "FAIL",
        f"Training and testing sets use the same {len(feature_names)} features in the same order.",
    )
    finite_values = np.isfinite(x_train_array).all() and np.isfinite(x_test_array).all()
    add_check(
        checks,
        "Finite model inputs",
        "PASS" if finite_values else "FAIL",
        "No missing or infinite values remain in the ANN model inputs.",
    )

    train_set.to_csv(DATA_DIR / "ann_training_set.csv", index=False)
    test_set.to_csv(DATA_DIR / "ann_testing_set.csv", index=False)
    pd.DataFrame({"Feature Order": range(1, len(feature_names) + 1), "Feature": feature_names}).to_csv(
        DOC_DIR / "ann_feature_names.csv", index=False
    )
    joblib.dump(preprocessor, PREPROCESSOR_DIR / "ann_preprocessor.joblib")

    class_rows = []
    for split_name, target in [("Full dataset", y), ("Training", y_train), ("Testing", y_test)]:
        for class_value, class_label in [(0, "No churn"), (1, "Churn")]:
            count = int((target == class_value).sum())
            class_rows.append(
                {
                    "Split": split_name,
                    "Class": class_label,
                    "Count": count,
                    "Percentage": round(count / len(target) * 100, 2),
                }
            )
    class_distribution = pd.DataFrame(class_rows)
    class_distribution.to_csv(DOC_DIR / "ann_class_distribution.csv", index=False)

    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)
    class_weights = pd.DataFrame(
        {
            "Class Value": [0, 1],
            "Class Label": ["No churn", "Churn"],
            "Suggested Balanced Weight": [round(float(weights[0]), 4), round(float(weights[1]), 4)],
        }
    )
    class_weights.to_csv(DOC_DIR / "ann_class_weights.csv", index=False)

    validation = pd.DataFrame(checks)
    validation.to_csv(DOC_DIR / "ann_data_validation_checks.csv", index=False)
    if (validation["Status"] == "FAIL").any():
        failed = validation.loc[validation["Status"] == "FAIL", "Check"].tolist()
        raise ValueError(f"ANN data validation failed: {failed}")

    train_churn_rate = float(y_train.mean())
    test_churn_rate = float(y_test.mean())
    metadata = pd.DataFrame(
        {
            "Item": [
                "Source rows",
                "Source columns",
                "Training rows",
                "Testing rows",
                "Input features after encoding",
                "Numeric columns scaled",
                "Categorical columns encoded",
                "Training churn rate",
                "Testing churn rate",
                "Exact duplicate rows retained",
                "Random state",
                "Test size",
            ],
            "Value": [
                len(df),
                df.shape[1],
                len(train_set),
                len(test_set),
                len(feature_names),
                ", ".join(numeric_columns),
                ", ".join(categorical_columns),
                f"{train_churn_rate:.2%}",
                f"{test_churn_rate:.2%}",
                exact_duplicate_rows,
                RANDOM_STATE,
                TEST_SIZE,
            ],
        }
    )
    metadata.to_csv(DOC_DIR / "ann_data_metadata.csv", index=False)

    summary = f"""# ANN Data Validation Summary

## Purpose

This document records the data validation and preprocessing handover for the Stage 3 artificial neural network. The main conclusion is that the data is ready for ANN training after using a training-only preprocessing pipeline that prevents test-set leakage.

## Responsibility

Pratima Kandel, Data Engineer, owns the model-ready dataset validation. Ranjit Mishra checks the deliverable against the assessment requirements. Muhammad Fahad Nazir uses these outputs for ANN architecture, training, and evaluation.

## Verified Result

- Source dataset: {len(df):,} rows and {df.shape[1]} columns.
- Missing values: {missing_values}.
- Training set: {len(train_set):,} rows.
- Testing set: {len(test_set):,} rows.
- Model input features after encoding: {len(feature_names)}.
- Training churn rate: {train_churn_rate:.2%}.
- Testing churn rate: {test_churn_rate:.2%}.
- Target column: `{TARGET_COLUMN}`, encoded as No = 0 and Yes = 1.
- Numeric scaling: `StandardScaler` fitted on training data only.
- Categorical encoding: `OneHotEncoder` fitted on training data only with unknown-category handling.

## Leakage Controls

The ANN preparation performs the train/test split before fitting the scaler or encoder. The fitted preprocessing pipeline is then used to transform the testing data. The target and any generated identifier are excluded from the model features. Training and testing columns are checked for the same names and order.

## Class Balance

The churn class represents {float(y.mean()):.2%} of the full dataset. Accuracy will not be used alone because the classes are not balanced. The training plan includes precision, recall, F1-score, ROC-AUC, a confusion matrix, and balanced class weights when needed.

## Data Limitation

The source contains {exact_duplicate_rows} identical rows ({duplicate_rate:.2%}). They are retained because the supplied data does not include a true customer identifier, so identical customer profiles cannot be proven to be accidental duplicate records. This limitation must be reported when interpreting model performance.

## Output Files

| File | Purpose |
| --- | --- |
| `data/ann_training_set.csv` | Leakage-safe ANN training data with the encoded target. |
| `data/ann_testing_set.csv` | ANN testing data transformed with the training-fitted pipeline. |
| `preprocessing/ann_preprocessor.joblib` | Saved encoder and scaler for reproducible predictions. |
| `docs/ann_feature_names.csv` | Ordered model input feature list. |
| `docs/ann_class_distribution.csv` | Class counts and percentages for full, training, and testing data. |
| `docs/ann_class_weights.csv` | Suggested balanced weights for ANN training. |
| `docs/ann_data_validation_checks.csv` | Pass and review evidence for each validation check. |
| `docs/ann_data_metadata.csv` | Dataset sizes and preprocessing configuration. |

## Handover Decision

The model-ready files are approved for ANN architecture and training. The duplicate-row limitation and class imbalance must remain visible in the final report and evaluation.
"""
    (DOC_DIR / "ann_data_validation_summary.md").write_text(summary, encoding="utf-8")

    print("ANN data preparation and validation completed.")
    print(validation.to_string(index=False))
    print(metadata.to_string(index=False))


if __name__ == "__main__":
    main()
