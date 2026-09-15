# Clustering Analysis Summary

## Purpose

This document explains the clustering analysis completed for Stage 2 of the Customer Churn Analysis project.

## Team Contribution

| Member | Role | Contribution |
| --- | --- | --- |
| Muhammad Fahad Nazir | Data Analyst - Clustering Analysis | Ran the elbow method, trained the K-Means model, labelled clusters, and prepared cluster visualisations. |
| Pratima Kandel | Data Engineer | Prepared the encoded and scaled dataset used for clustering. |
| SahilKumar Pravinbhai Patel | Business Analyst | Reviewed cluster profiles so the findings can support retention recommendations. |
| Ranjit Mishra | Project Manager | Organised outputs and checked that clustering deliverables match the Stage 2 assessment. |

## Method

K-Means clustering was used to segment customers. The `Churn` column was not used as an input feature for clustering. It was kept only for interpretation after the clusters were created.

The elbow method was tested for cluster counts from 2 to 8. The selected cluster number is `4` because it gives a simple and explainable grouping from the elbow result.

## Selected Cluster Result

- Selected number of clusters: `4`
- Inertia for selected cluster count: `24293.01`
- Silhouette score for selected cluster count: `0.172`

## Cluster Profile Summary

| Cluster | Customers | Churn Rate | Average Tenure | Average Monthly Charges | Senior Citizen Rate | Segment Label | Most Common Contract | Most Common Internet Service | Most Common Phone Service | Most Common Multiple Lines |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2058 | 0.448 | 12.474 | 73.752 | 0.0 | High-risk newer customers | Month-to-month | DSL | Yes | No |
| 1 | 2037 | 0.112 | 29.059 | 26.463 | 0.0 | Standard service customers | Month-to-month | DSL | Yes | No |
| 2 | 1806 | 0.135 | 58.195 | 88.183 | 0.0 | Loyal long-term customers | Month-to-month | DSL | Yes | No |
| 3 | 1142 | 0.417 | 33.296 | 79.817 | 1.0 | Higher-value service customers | Month-to-month | DSL | Yes | No |

## Output Files

| File | Purpose |
| --- | --- |
| `optimal_clusters_summary.csv` | Inertia and silhouette scores for each tested cluster number. |
| `cluster_profile_summary.csv` | Cluster size, churn rate, average tenure, monthly charges, and segment labels. |
| `customers_with_cluster_labels.csv` | Original customer records with assigned cluster labels. |
| `kmeans_model.pkl` | Saved trained K-Means model. |
| `elbow_method.png` | Visual evidence for selecting the optimal number of clusters. |
| `silhouette_scores.png` | Supporting visual for cluster quality. |
| `cluster_size.png` | Number of customers in each cluster. |
| `churn_rate_by_cluster.png` | Churn rate comparison across clusters. |
| `tenure_monthlycharges_clusters.png` | Simple scatter plot of customer segments. |

## Simple Interpretation

The clusters show that customers can be grouped by tenure, monthly charges, contract type, and service patterns. These groups help the team explain which customer segments may need retention attention.

## Limitation

K-Means is useful for segmentation, but it does not directly predict churn. The predictive modelling role is currently unassigned, so prediction work will be documented as a limitation unless the team later redistributes that work.
