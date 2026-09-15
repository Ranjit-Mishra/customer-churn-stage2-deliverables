# Data Preparation Summary

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
- Rows: 7043
- Columns: 10
- Target column: `Churn`
- Missing values found: 0

## Preparation Steps

1. Loaded the official course dataset.
2. Checked data types and missing values.
3. Added a simple `CustomerID` field for traceability.
4. Encoded categorical variables using one-hot encoding.
5. Scaled numeric fields using `StandardScaler`.
6. Split the data into training and testing sets using an 80/20 split.

## Scaling Technique

`StandardScaler` was used for numeric columns: SeniorCitizen, tenure, MonthlyCharges.

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
