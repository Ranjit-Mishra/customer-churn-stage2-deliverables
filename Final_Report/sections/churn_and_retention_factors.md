# Churn and Retention Factors

## Purpose

This section combines customer segmentation and ANN prediction evidence to identify the strongest churn and retention patterns. The findings support targeted action but do not prove that any single attribute causes churn.

## Segment Evidence

| Segment | Customers | Churn Rate | Average Tenure | Average Monthly Charges | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| High-risk newer customers | 2,058 | 44.8% | 12.5 months | $73.75 | Highest-risk segment; short tenure and month-to-month service indicate an early-life retention problem. |
| Standard service customers | 2,037 | 11.2% | 29.1 months | $26.46 | Lowest-cost segment with relatively low churn; broad discounts are unlikely to be efficient. |
| Loyal long-term customers | 1,806 | 13.5% | 58.2 months | $88.18 | Long tenure supports retention even with higher monthly charges. |
| Higher-value service customers | 1,142 | 41.7% | 33.3 months | $79.82 | High-value senior segment with high churn; affordability, service support and value perception require attention. |

## Predictive Evidence

The ANN achieved ROC-AUC 0.807, churn recall 0.693, churn precision 0.495 and churn F1-score 0.578. Recall met the project target, which means the model identifies a useful share of actual churn cases. Precision is lower, so model flags must support human review rather than trigger automatic offers.

The strongest permutation signals were tenure, MonthlyCharges, SeniorCitizen, Contract_Two year, PhoneService_No. Tenure and monthly charges were substantially more influential than the remaining inputs. This agrees with the segmentation evidence: newer customers and higher-charge groups require closer attention, while long tenure is associated with retention.

## Main Churn Factors

1. **Short tenure.** The highest-risk segment averages only 12.5 months of tenure and has 44.8% churn. Early service experience is therefore a priority.
2. **Higher monthly charges.** Monthly charges are the second strongest ANN signal. The two high-risk segments average $73.75 and $79.82 per month.
3. **Month-to-month relationships.** Month-to-month is the most common contract in every cluster. It provides flexibility but also makes departure easier, so contract-conversion offers should be tested for suitable customers.
4. **Senior-customer needs.** The higher-value senior segment has 41.7% churn. Billing clarity, accessible support and value review are important hypotheses to test.
5. **Service configuration.** Phone service, multiple lines and internet-service indicators appear in the model evidence, but their individual permutation impact is smaller than tenure and charges.

## Main Retention Factors

1. **Long tenure.** The loyal segment averages 58.2 months and has only 13.5% churn.
2. **Established service value.** The loyal segment remains despite average monthly charges of $88.18, suggesting that value and relationship length matter together.
3. **Lower-cost service.** The standard segment averages $26.46 and has 11.2% churn.

## Business Decision

Retention effort should be prioritised by combining ANN risk score, segment, tenure and customer value. The team should avoid one broad discount for all customers. Interventions should be tested, measured and reviewed for fairness and profitability.
