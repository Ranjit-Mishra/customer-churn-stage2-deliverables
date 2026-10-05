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
| `Predictive_Modeling/docs` | Data validation, class balance, feature list, architecture, training plan, and supporting evidence. |
| `scripts` | Python scripts used to reproduce Stage 2 outputs. |

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

## Key Result

The elbow method selected 4 clusters. These clusters were labelled and interpreted using customer tenure, monthly charges, senior citizen status, service type, contract type, and churn rate.

## Stage 3 ANN Readiness

The ANN data validation and architecture plan are complete:

- 5,634 training rows and 1,409 testing rows.
- 16 aligned model input features.
- No missing or infinite model inputs.
- Target and generated identifiers excluded from model features.
- Encoder and scaler fitted only on training data to prevent test-set leakage.
- Saved preprocessing pipeline, feature list, class distribution, and suggested class weights.
- Defined two-hidden-layer ANN architecture with dropout, early stopping, evaluation metrics, and measurable success targets.

The main evidence document is `Predictive_Modeling/docs/ANN_Data_Validation_and_Architecture_Plan.docx`.

## Current Limitation

The original predictive-modelling member is unavailable, so the four active members redistributed the Stage 3 work. The source dataset also contains 302 identical rows but no true customer identifier. These rows are retained because identical customer profiles cannot be proven to be accidental duplicates. This limitation will remain visible during model evaluation and final reporting.
