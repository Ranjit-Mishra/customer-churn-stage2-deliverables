from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
CLUSTER_PATH = BASE_DIR / "Clustering_Analysis" / "docs" / "cluster_profile_summary.csv"
METRICS_PATH = BASE_DIR / "Predictive_Modeling" / "results" / "ann_test_metrics.csv"
IMPORTANCE_PATH = BASE_DIR / "Predictive_Modeling" / "results" / "ann_permutation_feature_importance.csv"
OUT_DIR = BASE_DIR / "Final_Report"
SECTION_DIR = OUT_DIR / "sections"


def metric_value(metrics: pd.DataFrame, name: str) -> float:
    return float(metrics.loc[metrics["Metric"] == name, "Value"].iloc[0])


def main() -> None:
    SECTION_DIR.mkdir(parents=True, exist_ok=True)
    clusters = pd.read_csv(CLUSTER_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    importance = pd.read_csv(IMPORTANCE_PATH)

    c0 = clusters.loc[clusters["Cluster"] == 0].iloc[0]
    c1 = clusters.loc[clusters["Cluster"] == 1].iloc[0]
    c2 = clusters.loc[clusters["Cluster"] == 2].iloc[0]
    c3 = clusters.loc[clusters["Cluster"] == 3].iloc[0]
    top_features = importance.head(5)["Feature"].tolist()

    factors = f"""# Churn and Retention Factors

## Purpose

This section combines customer segmentation and ANN prediction evidence to identify the strongest churn and retention patterns. The findings support targeted action but do not prove that any single attribute causes churn.

## Segment Evidence

| Segment | Customers | Churn Rate | Average Tenure | Average Monthly Charges | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| {c0['Segment Label']} | {int(c0['Customers']):,} | {c0['Churn Rate']:.1%} | {c0['Average Tenure']:.1f} months | ${c0['Average Monthly Charges']:.2f} | Highest-risk segment; short tenure and month-to-month service indicate an early-life retention problem. |
| {c1['Segment Label']} | {int(c1['Customers']):,} | {c1['Churn Rate']:.1%} | {c1['Average Tenure']:.1f} months | ${c1['Average Monthly Charges']:.2f} | Lowest-cost segment with relatively low churn; broad discounts are unlikely to be efficient. |
| {c2['Segment Label']} | {int(c2['Customers']):,} | {c2['Churn Rate']:.1%} | {c2['Average Tenure']:.1f} months | ${c2['Average Monthly Charges']:.2f} | Long tenure supports retention even with higher monthly charges. |
| {c3['Segment Label']} | {int(c3['Customers']):,} | {c3['Churn Rate']:.1%} | {c3['Average Tenure']:.1f} months | ${c3['Average Monthly Charges']:.2f} | High-value senior segment with high churn; affordability, service support and value perception require attention. |

## Predictive Evidence

The ANN achieved ROC-AUC {metric_value(metrics, 'roc_auc'):.3f}, churn recall {metric_value(metrics, 'recall'):.3f}, churn precision {metric_value(metrics, 'precision'):.3f} and churn F1-score {metric_value(metrics, 'f1_score'):.3f}. Recall met the project target, which means the model identifies a useful share of actual churn cases. Precision is lower, so model flags must support human review rather than trigger automatic offers.

The strongest permutation signals were {', '.join(top_features)}. Tenure and monthly charges were substantially more influential than the remaining inputs. This agrees with the segmentation evidence: newer customers and higher-charge groups require closer attention, while long tenure is associated with retention.

## Main Churn Factors

1. **Short tenure.** The highest-risk segment averages only {c0['Average Tenure']:.1f} months of tenure and has {c0['Churn Rate']:.1%} churn. Early service experience is therefore a priority.
2. **Higher monthly charges.** Monthly charges are the second strongest ANN signal. The two high-risk segments average ${c0['Average Monthly Charges']:.2f} and ${c3['Average Monthly Charges']:.2f} per month.
3. **Month-to-month relationships.** Month-to-month is the most common contract in every cluster. It provides flexibility but also makes departure easier, so contract-conversion offers should be tested for suitable customers.
4. **Senior-customer needs.** The higher-value senior segment has {c3['Churn Rate']:.1%} churn. Billing clarity, accessible support and value review are important hypotheses to test.
5. **Service configuration.** Phone service, multiple lines and internet-service indicators appear in the model evidence, but their individual permutation impact is smaller than tenure and charges.

## Main Retention Factors

1. **Long tenure.** The loyal segment averages {c2['Average Tenure']:.1f} months and has only {c2['Churn Rate']:.1%} churn.
2. **Established service value.** The loyal segment remains despite average monthly charges of ${c2['Average Monthly Charges']:.2f}, suggesting that value and relationship length matter together.
3. **Lower-cost service.** The standard segment averages ${c1['Average Monthly Charges']:.2f} and has {c1['Churn Rate']:.1%} churn.

## Business Decision

Retention effort should be prioritised by combining ANN risk score, segment, tenure and customer value. The team should avoid one broad discount for all customers. Interventions should be tested, measured and reviewed for fairness and profitability.
"""
    (SECTION_DIR / "churn_and_retention_factors.md").write_text(factors, encoding="utf-8")

    recommendations = f"""# Targeted Retention Recommendations

## Decision

The company should use a tiered retention approach rather than a single campaign. ANN scores should rank customers for review, while segment information determines the most suitable intervention.

## Recommended Actions

| Priority Group | Evidence | Recommended Action | Measure |
| --- | --- | --- | --- |
| High-risk newer customers | {int(c0['Customers']):,} customers, {c0['Churn Rate']:.1%} churn, {c0['Average Tenure']:.1f}-month average tenure | Introduce a 30, 60 and 90-day onboarding journey, proactive service check, issue-resolution follow-up and suitable contract-conversion offer. | 90-day churn, issue resolution time, offer acceptance and retained revenue |
| Higher-value senior customers | {int(c3['Customers']):,} customers, {c3['Churn Rate']:.1%} churn, ${c3['Average_Monthly_Charges'] if 'Average_Monthly_Charges' in c3 else c3['Average Monthly Charges']:.2f} average monthly charges | Provide a clear bill review, accessible priority support, service-value check and price-protection or plan-right-sizing options. | Churn, repeat contacts, bill complaints, plan changes and satisfaction |
| Loyal long-term customers | {int(c2['Customers']):,} customers, {c2['Churn Rate']:.1%} churn, {c2['Average Tenure']:.1f}-month average tenure | Protect the relationship with recognition, loyalty benefits, referral options and relevant upgrades rather than blanket discounts. | Retention, referral uptake, upgrade acceptance and margin |
| Standard service customers | {int(c1['Customers']):,} customers and {c1['Churn Rate']:.1%} churn | Use low-cost digital education and service reminders. Escalate only when the ANN score rises or a service issue occurs. | Digital engagement, churn and intervention cost |
| ANN high-risk queue | ROC-AUC {metric_value(metrics, 'roc_auc'):.3f}, recall {metric_value(metrics, 'recall'):.3f}, precision {metric_value(metrics, 'precision'):.3f} | Rank customers for human review. Combine probability, segment and value before offering an incentive. | Precision, recall, retained revenue, incentive cost and false-positive rate |

## Implementation Sequence

1. Start a four-week pilot for the two high-risk segments.
2. Use the 0.50 ANN threshold as an initial ranking rule, not an automatic decision.
3. Let retention staff review recent issues, value and eligibility before contact.
4. Randomly hold out a comparable group so incremental retention impact can be measured.
5. Review churn reduction, revenue saved, offer cost, complaints and fairness before expansion.

## Governance

- Do not use age or senior status as the only reason for a decision.
- Explain offers clearly and provide consistent eligibility rules.
- Record contact outcomes and opt-outs.
- Monitor whether one group receives excessive contact or poorer offers.
- Retrain only after data quality and outcome labels are checked.

## Expected Business Value

The recommendations focus limited retention resources on customers with the clearest risk and value signals. Business value should be demonstrated through a controlled pilot rather than assumed from model accuracy alone.
"""
    (SECTION_DIR / "targeted_retention_recommendations.md").write_text(
        recommendations, encoding="utf-8"
    )

    limitations = f"""# Limitations and Proposed Solutions

| Limitation | Impact | Proposed Solution | Owner |
| --- | --- | --- | --- |
| Only nine customer attributes are supplied | Important causes such as complaints, payment behaviour, outages and usage changes are missing. | Add service quality, billing, support, payment and usage-history fields after governance review. | Data Engineer |
| No true customer identifier | The 302 identical rows cannot be confirmed as valid customers or accidental duplicates. | Add a stable anonymised customer key and define duplicate-resolution rules before future training. | Data Engineer |
| One historical dataset | Performance may change for future customers or another market. | Use a time-based holdout and monitor monthly performance after deployment. | ANN Technical Lead |
| Churn is the minority class | Accuracy can hide missed churn cases. | Continue class weighting and report recall, F1-score, balanced accuracy, ROC-AUC and precision-recall evidence. | ANN Technical Lead |
| Accuracy {metric_value(metrics, 'accuracy'):.3f} and F1 {metric_value(metrics, 'f1_score'):.3f} missed their project targets | The current model prioritises recall and produces more false alerts. | Tune architecture and threshold using validation data, collect stronger features and compare against simpler baselines. | ANN Technical Lead |
| Initial threshold is 0.50 | The threshold may not match the cost of missed churn and unnecessary offers. | Select a threshold using validation data and a documented cost-benefit rule. | Project Manager and Business Analyst |
| Permutation importance is not causal | A predictive signal does not prove why customers leave. | Combine model evidence with customer research, controlled pilots and complaint analysis. | Business Analyst |
| K-Means assumes compact distance-based groups | Real customer segments may overlap or have different shapes. | Compare with hierarchical or density-based clustering and review stability over time. | Clustering Analyst |
| No production monitoring plan has been tested | Performance and fairness could drift after implementation. | Define monitoring for data drift, recall, precision, segment outcomes, complaints and retraining triggers. | Project Manager |

## Overall Limitation Statement

The results are suitable for an educational proof of concept and decision support. They should not be treated as a production-ready automated decision system. A real deployment requires stronger data, privacy and fairness review, controlled testing, operational integration and ongoing monitoring.
"""
    (SECTION_DIR / "limitations_and_solutions.md").write_text(limitations, encoding="utf-8")

    outline = """# Final Report Draft Structure

1. Executive Summary
2. Business Problem and Project Objectives
3. Team Roles and Project Method
4. Data Preparation and Validation
5. Customer Segmentation Results
6. ANN Architecture and Training
7. ANN Performance and Prediction Findings
8. Churn and Retention Factors
9. Targeted Retention Recommendations
10. Limitations and Proposed Solutions
11. Conclusion and Next Steps
12. Evidence and Repository Links

The Project Manager will combine the verified Stage 2 evidence, ANN results and three completed business-analysis sections into the final report by 20 Oct 2026. Each member will review the section linked to their role before the final submission package is completed.
"""
    (OUT_DIR / "final_report_draft_structure.md").write_text(outline, encoding="utf-8")

    readme = """# Final Report Working Folder

This folder contains verified sections for the Stage 3 Final Report.

- `sections/churn_and_retention_factors.md`
- `sections/targeted_retention_recommendations.md`
- `sections/limitations_and_solutions.md`
- `final_report_draft_structure.md`

The completed report document and video evidence will be added after integration and review.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    print("Business analysis sections created.")
    print(f"Top ANN features: {', '.join(top_features)}")
    print(f"High-risk cluster churn rates: {c0['Churn Rate']:.1%}, {c3['Churn Rate']:.1%}")


if __name__ == "__main__":
    main()
