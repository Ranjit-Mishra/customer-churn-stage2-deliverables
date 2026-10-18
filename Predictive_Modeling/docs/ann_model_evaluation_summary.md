# ANN Model Training and Evaluation Summary

## Result

The Stage 3 artificial neural network was trained on 5,634 customer records and evaluated once on the reserved 1,409-record testing set. The model completed 19 epochs using early stopping.

## Implementation

- Model: scikit-learn `MLPClassifier`, a feed-forward artificial neural network.
- Architecture: 16 inputs, 32-neuron ReLU hidden layer, 16-neuron ReLU hidden layer, and binary probability output.
- Optimizer: Adam with learning rate 0.001.
- Overfitting controls: L2 regularisation at 0.001, 20% validation split, and early stopping with patience 10.
- Class imbalance: balanced sample weights were used during training.
- Decision threshold: 0.50.
- Controlled change: the implemented library does not expose dropout, so L2 regularisation replaced the planned dropout layers. Early stopping and validation monitoring were retained.

## Test Performance

| Metric | Value | Project Target | Status |
| --- | ---: | ---: | --- |
| Accuracy | 0.731 | 0.75 | Not met |
| Balanced accuracy | 0.719 | Not set | Reported |
| Churn precision | 0.495 | Not set | Reported |
| Churn recall | 0.693 | 0.65 | Met |
| Churn F1-score | 0.577 | 0.60 | Not met |
| ROC-AUC | 0.807 | 0.75 | Met |
| Average precision | 0.606 | Not set | Reported |

The model met 2 of the 4 measurable project targets.

## Confusion Matrix

- Correctly predicted no churn: 771.
- Incorrect churn warnings: 264.
- Missed churn cases: 115.
- Correctly identified churn cases: 259.

## Main Predictive Signals

The five strongest permutation signals were tenure, MonthlyCharges, SeniorCitizen, Contract_Two year, PhoneService_No. Permutation importance shows which model inputs most affected test-set ROC-AUC when shuffled. It supports interpretation but does not prove that a feature causes churn.

## Business Meaning

The model should be used to rank customers for retention attention rather than to make automatic decisions. Churn recall is important because a missed churn case represents a customer who may leave without intervention. False positives also matter because unnecessary incentives can reduce margin.

## Limitations

- The source has only nine customer attributes and does not include usage history, complaints, payment behaviour, service quality, or a true customer identifier.
- The source contains 302 identical rows that cannot be confirmed as accidental duplicates.
- The model is validated on one supplied dataset and has not been tested on future or external customers.
- Permutation importance is predictive, not causal.
- The 0.50 decision threshold is a starting point and should be reviewed against the cost of missed churn and unnecessary offers.

## Decision

The trained ANN, predictions, metrics, training history, feature importance and visualisations are ready for the final report and demonstration video. The next work is business interpretation, report integration and final submission review.
