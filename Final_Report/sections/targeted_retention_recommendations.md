# Targeted Retention Recommendations

## Decision

The company should use a tiered retention approach rather than a single campaign. ANN scores should rank customers for review, while segment information determines the most suitable intervention.

## Recommended Actions

| Priority Group | Evidence | Recommended Action | Measure |
| --- | --- | --- | --- |
| High-risk newer customers | 2,058 customers, 44.8% churn, 12.5-month average tenure | Introduce a 30, 60 and 90-day onboarding journey, proactive service check, issue-resolution follow-up and suitable contract-conversion offer. | 90-day churn, issue resolution time, offer acceptance and retained revenue |
| Higher-value senior customers | 1,142 customers, 41.7% churn, $79.82 average monthly charges | Provide a clear bill review, accessible priority support, service-value check and price-protection or plan-right-sizing options. | Churn, repeat contacts, bill complaints, plan changes and satisfaction |
| Loyal long-term customers | 1,806 customers, 13.5% churn, 58.2-month average tenure | Protect the relationship with recognition, loyalty benefits, referral options and relevant upgrades rather than blanket discounts. | Retention, referral uptake, upgrade acceptance and margin |
| Standard service customers | 2,037 customers and 11.2% churn | Use low-cost digital education and service reminders. Escalate only when the ANN score rises or a service issue occurs. | Digital engagement, churn and intervention cost |
| ANN high-risk queue | ROC-AUC 0.807, recall 0.693, precision 0.495 | Rank customers for human review. Combine probability, segment and value before offering an incentive. | Precision, recall, retained revenue, incentive cost and false-positive rate |

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
