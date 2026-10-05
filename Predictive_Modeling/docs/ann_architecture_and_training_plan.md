# ANN Architecture and Training Plan

## Purpose and Decision

This plan defines the Stage 3 artificial neural network for predicting customer churn. The selected model is a compact binary classifier with two hidden layers, dropout, early stopping, and class weighting. It is intentionally simple enough to explain and reproduce while still supporting non-linear relationships in the customer data.

## Responsibility

Muhammad Fahad Nazir is the ANN technical lead and owns the architecture, training, prediction, and evaluation evidence. Pratima Kandel supplies the validated model-ready data. Ranjit Mishra tracks progress and checks the assessment evidence. SahilKumar Pravinbhai Patel uses the model findings for churn factors and retention recommendations.

## Validated Inputs

| Item | Value |
| --- | ---: |
| Training rows | 5,634 |
| Testing rows | 1,409 |
| Input features | 16 |
| Target | Churn, encoded as 0 or 1 |
| Training preprocessing | Standard scaling and one-hot encoding fitted on training data only |
| Suggested class weight for no churn | 0.6806 |
| Suggested class weight for churn | 1.8843 |

## Model Architecture

| Layer | Configuration | Purpose |
| --- | --- | --- |
| Input | 16 model features | Receives the validated customer attributes. |
| Dense 1 | 32 neurons, ReLU | Learns the main non-linear feature relationships. |
| Dropout 1 | Rate 0.20 | Reduces overfitting during training. |
| Dense 2 | 16 neurons, ReLU | Learns a smaller combined representation. |
| Dropout 2 | Rate 0.20 | Adds a second overfitting control. |
| Output | 1 neuron, Sigmoid | Produces a churn probability from 0 to 1. |

Architecture flow:

`16 inputs -> Dense 32 ReLU -> Dropout 0.20 -> Dense 16 ReLU -> Dropout 0.20 -> Sigmoid output`

## Compilation Settings

- Optimizer: Adam with learning rate 0.001.
- Loss function: binary cross-entropy.
- Training metrics: accuracy, precision, recall, and ROC-AUC.
- Random seed: 42 for reproducibility.

## Training Controls

- Maximum epochs: 100.
- Batch size: 32.
- Validation split: 20% of the training set.
- Early stopping: monitor validation loss with patience of 10 epochs.
- Best weights: restore the best validation-loss weights.
- Class weighting: use the balanced weights above because churn is the minority class.
- Testing data: use once after training decisions are complete.

## Evaluation Evidence

The saved evidence must include the trained model, prediction probabilities, predicted classes, training history, learning curves, confusion matrix, and a metrics table. The final evaluation will report accuracy, balanced accuracy, precision, recall, F1-score, and ROC-AUC. Accuracy will not be used alone.

## Measurable Success Targets

These are project targets, not guaranteed results:

- ROC-AUC of at least 0.75.
- Churn recall of at least 0.65.
- Churn F1-score of at least 0.60.
- Accuracy of at least 0.75.
- No evidence of severe overfitting, based on training and validation loss curves.

If a target is missed, the result will still be reported honestly. The team will review the threshold, class weights, hidden-layer size, dropout rate, and early-stopping behaviour without using the testing set for repeated tuning.

## Risks and Controls

| Risk | Control |
| --- | --- |
| Class imbalance makes accuracy misleading | Use class weights and report recall, F1-score, balanced accuracy, and ROC-AUC. |
| Model overfits the training data | Use validation data, dropout, early stopping, and learning curves. |
| Test-set leakage inflates performance | Fit preprocessing on training data only and reserve testing data for final evaluation. |
| Results are difficult to reproduce | Save the preprocessing pipeline, configuration, random seed, code, model, and metrics. |
| Identical source rows influence results | Retain and disclose them because the source has no true customer ID; document this limitation. |

## Completion Checklist

- [ ] Build the ANN from `ann_architecture_config.json`.
- [ ] Train with early stopping and class weights.
- [ ] Save the trained model and training history.
- [ ] Predict churn probabilities for the testing set.
- [ ] Create the metrics table, confusion matrix, ROC curve, and learning curves.
- [ ] Compare results with the measurable success targets.
- [ ] Give the model evidence to the Project Manager and Business Analyst.

## Handover Decision

The architecture and training plan are approved for implementation after the ANN-ready data validation. Training and final evaluation are the next dependent tasks.
