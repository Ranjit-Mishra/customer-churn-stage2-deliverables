# Customer Churn Analysis Stage 2 Deliverables

This repository contains the Stage 2 submission package for the Customer Churn Analysis project for Advanced Telecommunications Solutions.

## Team

| Member | Role | Stage 2 Contribution |
| --- | --- | --- |
| Ranjit Mishra | Project Manager | Organised the repository, checked assessment requirements, coordinated outputs, and prepared submission materials. |
| Pratima Kandel | Data Engineer | Prepared the dataset, checked missing values, encoded categorical variables, scaled numeric columns, and created training/testing sets. |
| Muhammad Fahad Nazir | Data Analyst - Clustering Analysis | Ran the elbow method, trained the K-Means model, labelled clusters, and prepared visualisations. |
| SahilKumar Pravinbhai Patel | Business Analyst | Reviewed cluster profiles and linked findings to churn interpretation and future retention recommendations. |

## Repository Structure

| Folder | Purpose |
| --- | --- |
| `Data_Preparation/raw` | Original course dataset. |
| `Data_Preparation/processed` | Cleaned, encoded, scaled, and train/test output files. |
| `Data_Preparation/docs` | Data preparation summary and scaling documentation. |
| `Clustering_Analysis/models` | Saved trained K-Means model. |
| `Clustering_Analysis/visualisations` | Elbow chart, silhouette chart, cluster size chart, churn rate chart, and cluster scatter plot. |
| `Clustering_Analysis/docs` | Cluster profiles, labelled customer records, and clustering summary. |
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

## Limitation

The predictive modelling role is currently unassigned because the team has four active members. Stage 2 therefore focuses on data preparation and clustering analysis, as required by the S2W8A2 Stage 2 Deliverables assessment.
