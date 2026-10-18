# Limitations and Proposed Solutions

| Limitation | Impact | Proposed Solution | Owner |
| --- | --- | --- | --- |
| Only nine customer attributes are supplied | Important causes such as complaints, payment behaviour, outages and usage changes are missing. | Add service quality, billing, support, payment and usage-history fields after governance review. | Data Engineer |
| No true customer identifier | The 302 identical rows cannot be confirmed as valid customers or accidental duplicates. | Add a stable anonymised customer key and define duplicate-resolution rules before future training. | Data Engineer |
| One historical dataset | Performance may change for future customers or another market. | Use a time-based holdout and monitor monthly performance after deployment. | ANN Technical Lead |
| Churn is the minority class | Accuracy can hide missed churn cases. | Continue class weighting and report recall, F1-score, balanced accuracy, ROC-AUC and precision-recall evidence. | ANN Technical Lead |
| Accuracy 0.731 and F1 0.578 missed their project targets | The current model prioritises recall and produces more false alerts. | Tune architecture and threshold using validation data, collect stronger features and compare against simpler baselines. | ANN Technical Lead |
| Initial threshold is 0.50 | The threshold may not match the cost of missed churn and unnecessary offers. | Select a threshold using validation data and a documented cost-benefit rule. | Project Manager and Business Analyst |
| Permutation importance is not causal | A predictive signal does not prove why customers leave. | Combine model evidence with customer research, controlled pilots and complaint analysis. | Business Analyst |
| K-Means assumes compact distance-based groups | Real customer segments may overlap or have different shapes. | Compare with hierarchical or density-based clustering and review stability over time. | Clustering Analyst |
| No production monitoring plan has been tested | Performance and fairness could drift after implementation. | Define monitoring for data drift, recall, precision, segment outcomes, complaints and retraining triggers. | Project Manager |

## Overall Limitation Statement

The results are suitable for an educational proof of concept and decision support. They should not be treated as a production-ready automated decision system. A real deployment requires stronger data, privacy and fairness review, controlled testing, operational integration and ongoing monitoring.
