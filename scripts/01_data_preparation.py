from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "Data_Preparation" / "raw" / "Dataset_ATS_v2.csv"
OUT_DIR = BASE_DIR / "Data_Preparation" / "processed"
DOC_DIR = BASE_DIR / "Data_Preparation" / "docs"

TARGET_COLUMN = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(RAW_PATH)
    df = raw_df.copy()
    df.insert(0, "CustomerID", range(1, len(df) + 1))

    missing_summary = raw_df.isna().sum().reset_index()
    missing_summary.columns = ["Column", "Missing Values"]
    missing_summary["Missing Percentage"] = (
        missing_summary["Missing Values"] / len(raw_df) * 100
    ).round(2)

    y = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    features = df.drop(columns=[TARGET_COLUMN])
    customer_ids = features["CustomerID"]
    features_for_model = features.drop(columns=["CustomerID"])

    categorical_columns = features_for_model.select_dtypes(include=["str", "object"]).columns.tolist()
    numeric_columns = features_for_model.select_dtypes(include=["int64", "float64"]).columns.tolist()

    encoded_features = pd.get_dummies(
        features_for_model,
        columns=categorical_columns,
        drop_first=False,
        dtype=int,
    )

    scaler = StandardScaler()
    scaled_features = encoded_features.copy()
    scaled_features[numeric_columns] = scaler.fit_transform(encoded_features[numeric_columns])
    scaled_features.insert(0, "CustomerID", customer_ids)

    preprocessed_df = scaled_features.copy()
    preprocessed_df[TARGET_COLUMN] = y

    x_train, x_test, y_train, y_test = train_test_split(
        scaled_features,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_set = x_train.copy()
    train_set[TARGET_COLUMN] = y_train
    test_set = x_test.copy()
    test_set[TARGET_COLUMN] = y_test

    raw_df.to_csv(OUT_DIR / "cleaned_dataset.csv", index=False)
    preprocessed_df.to_csv(OUT_DIR / "preprocessed_dataset.csv", index=False)
    train_set.to_csv(OUT_DIR / "training_set.csv", index=False)
    test_set.to_csv(OUT_DIR / "testing_set.csv", index=False)
    missing_summary.to_csv(OUT_DIR / "missing_values_summary.csv", index=False)

    metadata = pd.DataFrame(
        {
            "Item": [
                "Rows in original dataset",
                "Columns in original dataset",
                "Rows in training set",
                "Rows in testing set",
                "Test size",
                "Target column",
                "Categorical columns encoded",
                "Numeric columns scaled",
                "Scaler used",
                "Random state",
            ],
            "Value": [
                len(raw_df),
                raw_df.shape[1],
                len(train_set),
                len(test_set),
                TEST_SIZE,
                TARGET_COLUMN,
                ", ".join(categorical_columns),
                ", ".join(numeric_columns),
                "StandardScaler",
                RANDOM_STATE,
            ],
        }
    )
    metadata.to_csv(OUT_DIR / "data_preparation_metadata.csv", index=False)

    summary_md = f"""# Data Preparation Summary

## Purpose

This document explains the data preparation work completed for Stage 2 of the Customer Churn Analysis project.

## Team Contribution

| Member | Role | Contribution |
| --- | --- | --- |
| Pratima Kandel | Data Engineer | Prepared the dataset, checked missing values, encoded categorical variables, scaled numeric columns, and created training/testing sets. |
| Ranjit Mishra | Project Manager | Organised the repository structure and checked that the deliverables match the assessment instructions. |
| Muhammad Fahad Nazir | Data Analyst - Clustering | Confirmed that the prepared dataset is suitable for clustering analysis. |
| SahilKumar Pravinbhai Patel | Business Analyst | Confirmed that the prepared attributes can support later churn interpretation and retention recommendations. |

## Dataset Overview

- Dataset file: `Dataset_ATS_v2.csv`
- Rows: {len(raw_df)}
- Columns: {raw_df.shape[1]}
- Target column: `{TARGET_COLUMN}`
- Missing values found: {int(raw_df.isna().sum().sum())}

## Preparation Steps

1. Loaded the official course dataset.
2. Checked data types and missing values.
3. Added a simple `CustomerID` field for traceability.
4. Encoded categorical variables using one-hot encoding.
5. Scaled numeric fields using `StandardScaler`.
6. Split the data into training and testing sets using an 80/20 split.

## Scaling Technique

`StandardScaler` was used for numeric columns: {", ".join(numeric_columns)}.

Standard scaling changes each numeric column so it has a mean close to 0 and a standard deviation close to 1. This helps K-Means because K-Means uses distance. Without scaling, a column with larger numbers can unfairly dominate the cluster result.

## Output Files

| File | Purpose |
| --- | --- |
| `cleaned_dataset.csv` | Original dataset checked and saved after confirming no missing values. |
| `preprocessed_dataset.csv` | Encoded and scaled dataset for analysis. |
| `training_set.csv` | 80% training split for model validation needs. |
| `testing_set.csv` | 20% testing split for model validation needs. |
| `missing_values_summary.csv` | Missing value count and percentage for each column. |
| `data_preparation_metadata.csv` | Summary of preprocessing choices and dataset sizes. |

## Limitation

The dataset is already clean and has only selected customer attributes. The predictive modelling role is currently unassigned, so Stage 2 focuses on data preparation and clustering deliverables.
"""

    (DOC_DIR / "data_preparation_summary.md").write_text(summary_md, encoding="utf-8")

    print("Data preparation completed.")
    print(metadata.to_string(index=False))


if __name__ == "__main__":
    main()
