# ANN Data Validation Summary

## Purpose

This document records the data validation and preprocessing handover for the Stage 3 artificial neural network. The main conclusion is that the data is ready for ANN training after using a training-only preprocessing pipeline that prevents test-set leakage.

## Responsibility

Pratima Kandel, Data Engineer, owns the model-ready dataset validation. Ranjit Mishra checks the deliverable against the assessment requirements. Muhammad Fahad Nazir uses these outputs for ANN architecture, training, and evaluation.

## Verified Result

- Source dataset: 7,043 rows and 10 columns.
- Missing values: 0.
- Training set: 5,634 rows.
- Testing set: 1,409 rows.
- Model input features after encoding: 16.
- Training churn rate: 26.54%.
- Testing churn rate: 26.54%.
- Target column: `Churn`, encoded as No = 0 and Yes = 1.
- Numeric scaling: `StandardScaler` fitted on training data only.
- Categorical encoding: `OneHotEncoder` fitted on training data only with unknown-category handling.

## Leakage Controls

The ANN preparation performs the train/test split before fitting the scaler or encoder. The fitted preprocessing pipeline is then used to transform the testing data. The target and any generated identifier are excluded from the model features. Training and testing columns are checked for the same names and order.

## Class Balance

The churn class represents 26.54% of the full dataset. Accuracy will not be used alone because the classes are not balanced. The training plan includes precision, recall, F1-score, ROC-AUC, a confusion matrix, and balanced class weights when needed.

## Data Limitation

The source contains 302 identical rows (4.29%). They are retained because the supplied data does not include a true customer identifier, so identical customer profiles cannot be proven to be accidental duplicate records. This limitation must be reported when interpreting model performance.

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
