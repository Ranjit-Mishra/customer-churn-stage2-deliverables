# Customer Churn Analysis Project Deliverables

This repository contains the completed Stage 2 submission and the developing Stage 3 predictive-modelling deliverables for the Customer Churn Analysis project for Advanced Telecommunications Solutions.

## Team

| Member | Role | Project Contribution |
| --- | --- | --- |
| Ranjit Mishra | Project Manager | Organises the repository, checks assessment requirements, coordinates outputs, maintains project evidence, integrates the final report, and prepares submission materials. |
| Pratima Kandel | Data Engineer | Prepared the Stage 2 data and validates the Stage 3 ANN inputs, leakage controls, class balance, feature order, and preprocessing pipeline. |
| Muhammad Fahad Nazir | Data Analyst - Clustering and ANN Technical Lead | Completed the clustering analysis and leads the Stage 3 ANN architecture, training, prediction, and evaluation. |
| SahilKumar Pravinbhai Patel | Business Analyst | Reviews clustering and ANN evidence and connects the findings to churn factors, retention recommendations, limitations, and business value. |

## Repository Structure

| Folder | Purpose |
| --- | --- |
| `Data_Preparation/raw` | Original course dataset. |
| `Data_Preparation/processed` | Cleaned, encoded, scaled, and train/test output files. |
| `Data_Preparation/docs` | Data preparation summary and scaling documentation. |
| `Clustering_Analysis/models` | Saved trained K-Means model. |
| `Clustering_Analysis/visualisations` | Elbow chart, silhouette chart, cluster size chart, churn rate chart, and cluster scatter plot. |
| `Clustering_Analysis/docs` | Cluster profiles, labelled customer records, and clustering summary. |
| `Predictive_Modeling/data` | ANN-ready training and testing files. |
| `Predictive_Modeling/preprocessing` | Saved training-fitted encoder and scaler pipeline. |
| `Predictive_Modeling/docs` | Data validation, architecture, training plan, evaluation summary, and supporting evidence. |
| `Predictive_Modeling/models` | Saved trained artificial neural network model. |
| `Predictive_Modeling/results` | Predictions, metrics, confusion matrix, training history, and feature importance. |
| `Predictive_Modeling/visualisations` | ANN learning, performance, and interpretation charts. |
| `Final_Report` | Draft report structure and business-analysis sections for Stage 3. |
| `scripts` | Python scripts used to reproduce and verify the project outputs. |

## How to Run

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

Run the scripts in this order:

```bash
python scripts/01_data_preparation.py
python scripts/02_clustering_analysis.py
python scripts/03_prepare_ann_data.py
python scripts/04_create_ann_architecture_plan.py
python scripts/05_build_ann_readiness_document.py
python scripts/06_verify_ann_readiness.py
python scripts/07_train_ann_model.py
python scripts/08_create_business_analysis.py
python scripts/09_verify_stage3_progress.py
```

## Main Outputs

Data Preparation:

- `Data_Preparation/processed/preprocessed_dataset.csv`
- `Data_Preparation/processed/training_set.csv`
- `Data_Preparation/processed/testing_set.csv`
- `Data_Preparation/docs/Data_Preparation_Documentation.docx`

Clustering Analysis:

- `Clustering_Analysis/docs/optimal_clusters_summary.csv`
- `Clustering_Analysis/models/kmeans_model.pkl`
- `Clustering_Analysis/docs/customers_with_cluster_labels.csv`
- `Clustering_Analysis/docs/cluster_profile_summary.csv`
- `Clustering_Analysis/visualisations/elbow_method.png`
- `Clustering_Analysis/visualisations/tenure_monthlycharges_clusters.png`
- `Clustering_Analysis/docs/Clustering_Analysis_Documentation.docx`

Predictive Modeling:

- `Predictive_Modeling/models/ann_mlp_model.joblib`
- `Predictive_Modeling/results/ann_test_metrics.csv`
- `Predictive_Modeling/results/ann_test_predictions.csv`
- `Predictive_Modeling/results/ann_confusion_matrix.csv`
- `Predictive_Modeling/results/ann_permutation_feature_importance.csv`
- `Predictive_Modeling/visualisations/ann_roc_curve.png`
- `Predictive_Modeling/visualisations/ann_confusion_matrix.png`
- `Predictive_Modeling/docs/ann_model_evaluation_summary.md`

Final Report Development:

- `Final_Report/sections/churn_and_retention_factors.md`
- `Final_Report/sections/targeted_retention_recommendations.md`
- `Final_Report/sections/limitations_and_solutions.md`
- `Final_Report/final_report_draft_structure.md`

## Key Result

The elbow method selected 4 clusters. These clusters were labelled and interpreted using customer tenure, monthly charges, senior citizen status, service type, contract type, and churn rate.

## Stage 3 ANN Results

The ANN data validation, architecture, training and evaluation are complete:

- 5,634 training rows and 1,409 testing rows.
- 16 aligned model input features.
- No missing or infinite model inputs.
- Target and generated identifiers excluded from model features.
- Encoder and scaler fitted only on training data to prevent test-set leakage.
- Saved preprocessing pipeline, feature list, class distribution, and suggested class weights.
- Implemented a two-hidden-layer feed-forward ANN with L2 regularisation, class-balanced training and early stopping.
- Evaluated 1,409 reserved test records after 19 training epochs.
- Achieved accuracy 0.731, balanced accuracy 0.719, churn recall 0.693, churn F1-score 0.578 and ROC-AUC 0.807.
- Met the project targets for churn recall and ROC-AUC; accuracy and F1-score remain below target and are documented as limitations.
- Generated predictions, confusion matrix, learning charts, ROC and precision-recall curves, and permutation feature importance.

The main evidence is in `Predictive_Modeling/docs/ANN_Data_Validation_and_Architecture_Plan.docx` and `Predictive_Modeling/docs/ann_model_evaluation_summary.md`.

## Current Status and Limitations

The original predictive-modelling member is unavailable, so the four active members redistributed the Stage 3 work. The source dataset also contains 302 identical rows but no true customer identifier. These rows are retained because identical customer profiles cannot be proven to be accidental duplicates.

The model is a decision-support proof of concept, not a production system. Its 0.50 threshold has not been cost-optimised, predictive importance is not causal, and performance has not been validated on future or external customers. These limits and practical solutions are recorded in the final-report sections.
