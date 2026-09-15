from pathlib import Path
import os

import joblib
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "Data_Preparation" / "raw" / "Dataset_ATS_v2.csv"
PROCESSED_PATH = BASE_DIR / "Data_Preparation" / "processed" / "preprocessed_dataset.csv"
OUT_DIR = BASE_DIR / "Clustering_Analysis"
MODEL_DIR = OUT_DIR / "models"
VIS_DIR = OUT_DIR / "visualisations"
DOC_DIR = OUT_DIR / "docs"

RANDOM_STATE = 42
K_RANGE = range(2, 9)


def choose_elbow_k(scores: pd.DataFrame) -> int:
    points = scores[["k", "inertia"]].to_numpy(dtype=float)
    first = points[0]
    last = points[-1]
    line = last - first
    line_norm = (line**2).sum() ** 0.5
    distances = []
    for point in points:
        if line_norm == 0:
            distances.append(0)
        else:
            distances.append(abs(line[0] * (first[1] - point[1]) - (first[0] - point[0]) * line[1]) / line_norm)
    scores = scores.copy()
    scores["elbow_distance"] = distances
    return int(scores.loc[scores["elbow_distance"].idxmax(), "k"])


def segment_name(row: pd.Series) -> str:
    if row["Churn Rate"] >= 0.40 and row["Average Tenure"] <= 25:
        return "High-risk newer customers"
    if row["Average Tenure"] >= 45 and row["Churn Rate"] <= 0.25:
        return "Loyal long-term customers"
    if row["Average Monthly Charges"] >= 65:
        return "Higher-value service customers"
    return "Standard service customers"


def save_bar_chart(data: pd.DataFrame, x: str, y: str, title: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    bars = plt.bar(data[x].astype(str), data[y], color="#2E74B5")
    plt.title(title)
    plt.xlabel("Cluster")
    plt.ylabel(ylabel)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(rows)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(RAW_PATH)
    raw_df.insert(0, "CustomerID", range(1, len(raw_df) + 1))
    processed_df = pd.read_csv(PROCESSED_PATH)

    feature_columns = [c for c in processed_df.columns if c not in ["CustomerID", "Churn"]]
    x = processed_df[feature_columns]

    score_rows = []
    for k in K_RANGE:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(x)
        score_rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette_score": silhouette_score(x, labels),
            }
        )

    scores = pd.DataFrame(score_rows)
    selected_k = choose_elbow_k(scores)

    final_model = KMeans(n_clusters=selected_k, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = final_model.fit_predict(x)

    clustered = raw_df.copy()
    clustered["Churn_Binary"] = clustered["Churn"].map({"No": 0, "Yes": 1})
    clustered["Cluster"] = cluster_labels

    profile = (
        clustered.groupby("Cluster")
        .agg(
            Customers=("CustomerID", "count"),
            Churn_Rate=("Churn_Binary", "mean"),
            Average_Tenure=("tenure", "mean"),
            Average_Monthly_Charges=("MonthlyCharges", "mean"),
            Senior_Citizen_Rate=("SeniorCitizen", "mean"),
        )
        .reset_index()
    )
    profile.columns = [
        "Cluster",
        "Customers",
        "Churn Rate",
        "Average Tenure",
        "Average Monthly Charges",
        "Senior Citizen Rate",
    ]
    for col in ["Churn Rate", "Average Tenure", "Average Monthly Charges", "Senior Citizen Rate"]:
        profile[col] = profile[col].round(3)
    profile["Segment Label"] = profile.apply(segment_name, axis=1)

    dominant = []
    for cluster, group in clustered.groupby("Cluster"):
        dominant.append(
            {
                "Cluster": cluster,
                "Most Common Contract": group["Contract"].mode().iloc[0],
                "Most Common Internet Service": group["InternetService"].mode().iloc[0],
                "Most Common Phone Service": group["PhoneService"].mode().iloc[0],
                "Most Common Multiple Lines": group["MultipleLines"].mode().iloc[0],
            }
        )
    dominant_df = pd.DataFrame(dominant)
    profile = profile.merge(dominant_df, on="Cluster", how="left")

    scores.to_csv(DOC_DIR / "optimal_clusters_summary.csv", index=False)
    profile.to_csv(DOC_DIR / "cluster_profile_summary.csv", index=False)
    clustered.to_csv(DOC_DIR / "customers_with_cluster_labels.csv", index=False)
    joblib.dump(final_model, MODEL_DIR / "kmeans_model.pkl")

    plt.figure(figsize=(8, 5))
    plt.plot(scores["k"], scores["inertia"], marker="o", color="#2E74B5")
    plt.axvline(selected_k, color="#C00000", linestyle="--", label=f"Selected k = {selected_k}")
    plt.title("Elbow Method for K-Means")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(VIS_DIR / "elbow_method.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(scores["k"], scores["silhouette_score"], marker="o", color="#548235")
    plt.axvline(selected_k, color="#C00000", linestyle="--", label=f"Selected k = {selected_k}")
    plt.title("Silhouette Scores by Cluster Count")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(VIS_DIR / "silhouette_scores.png", dpi=160)
    plt.close()

    save_bar_chart(profile, "Cluster", "Customers", "Customer Count by Cluster", "Customers", VIS_DIR / "cluster_size.png")
    save_bar_chart(profile, "Cluster", "Churn Rate", "Churn Rate by Cluster", "Churn rate", VIS_DIR / "churn_rate_by_cluster.png")

    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(
        clustered["tenure"],
        clustered["MonthlyCharges"],
        c=clustered["Cluster"],
        cmap="tab10",
        alpha=0.65,
        s=20,
    )
    plt.title("Customer Segments by Tenure and Monthly Charges")
    plt.xlabel("Tenure")
    plt.ylabel("Monthly Charges")
    handles, _ = scatter.legend_elements(prop="colors")
    plt.legend(handles, [f"Cluster {i}" for i in sorted(clustered["Cluster"].unique())], title="Cluster")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "tenure_monthlycharges_clusters.png", dpi=160)
    plt.close()

    selected_row = scores[scores["k"] == selected_k].iloc[0]
    summary_md = f"""# Clustering Analysis Summary

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

The elbow method was tested for cluster counts from 2 to 8. The selected cluster number is `{selected_k}` because it gives a simple and explainable grouping from the elbow result.

## Selected Cluster Result

- Selected number of clusters: `{selected_k}`
- Inertia for selected cluster count: `{selected_row["inertia"]:.2f}`
- Silhouette score for selected cluster count: `{selected_row["silhouette_score"]:.3f}`

## Cluster Profile Summary

{dataframe_to_markdown(profile)}

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
"""

    (DOC_DIR / "clustering_analysis_summary.md").write_text(summary_md, encoding="utf-8")

    print("Clustering analysis completed.")
    print(f"Selected k: {selected_k}")
    print(scores.to_string(index=False))
    print(profile.to_string(index=False))


if __name__ == "__main__":
    main()
