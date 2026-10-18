import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier


PM_DIR = BASE_DIR / "Predictive_Modeling"
DATA_DIR = PM_DIR / "data"
DOC_DIR = PM_DIR / "docs"
MODEL_DIR = PM_DIR / "models"
RESULT_DIR = PM_DIR / "results"
VIS_DIR = PM_DIR / "visualisations"

RANDOM_STATE = 42
TARGET = "Churn"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_DIR / "ann_training_set.csv")
    test = pd.read_csv(DATA_DIR / "ann_testing_set.csv")
    config = json.loads((DOC_DIR / "ann_architecture_config.json").read_text(encoding="utf-8"))
    class_weights_df = pd.read_csv(DOC_DIR / "ann_class_weights.csv")
    class_weights = dict(
        zip(
            class_weights_df["Class Value"].astype(int),
            class_weights_df["Suggested Balanced Weight"].astype(float),
        )
    )

    x_train = train.drop(columns=[TARGET])
    y_train = train[TARGET].astype(int)
    x_test = test.drop(columns=[TARGET])
    y_test = test[TARGET].astype(int)
    sample_weights = y_train.map(class_weights).to_numpy()

    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=32,
        learning_rate_init=0.001,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.20,
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
        verbose=False,
    )

    started = time.perf_counter()
    model.fit(x_train, y_train, sample_weight=sample_weights)
    training_seconds = time.perf_counter() - started

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(y_test, probabilities),
        "majority_class_baseline_accuracy": max(y_test.mean(), 1 - y_test.mean()),
    }

    target_map = {
        "accuracy": config["evaluation"]["success_targets"]["accuracy_minimum"],
        "recall": config["evaluation"]["success_targets"]["churn_recall_minimum"],
        "f1_score": config["evaluation"]["success_targets"]["churn_f1_minimum"],
        "roc_auc": config["evaluation"]["success_targets"]["roc_auc_minimum"],
    }
    metric_rows = []
    for name, value in metrics.items():
        target = target_map.get(name)
        metric_rows.append(
            {
                "Metric": name,
                "Value": round(float(value), 4),
                "Project Target": "Not set" if target is None else round(float(target), 4),
                "Target Status": "Not assessed" if target is None else ("Met" if value >= target else "Not met"),
            }
        )
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(RESULT_DIR / "ann_test_metrics.csv", index=False)

    report = classification_report(
        y_test,
        predictions,
        target_names=["No churn", "Churn"],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().reset_index(names="Class or Average").to_csv(
        RESULT_DIR / "ann_classification_report.csv", index=False
    )

    cm = confusion_matrix(y_test, predictions)
    pd.DataFrame(
        cm,
        index=["Actual No Churn", "Actual Churn"],
        columns=["Predicted No Churn", "Predicted Churn"],
    ).to_csv(RESULT_DIR / "ann_confusion_matrix.csv")

    prediction_output = pd.DataFrame(
        {
            "Test_Row": np.arange(1, len(y_test) + 1),
            "Actual_Churn": y_test.to_numpy(),
            "Predicted_Probability": np.round(probabilities, 6),
            "Predicted_Churn": predictions,
            "Correct_Prediction": predictions == y_test.to_numpy(),
        }
    )
    prediction_output.to_csv(RESULT_DIR / "ann_test_predictions.csv", index=False)

    history_rows = []
    validation_scores = list(getattr(model, "validation_scores_", []) or [])
    for index, loss in enumerate(model.loss_curve_, start=1):
        history_rows.append(
            {
                "Epoch": index,
                "Training Loss": round(float(loss), 6),
                "Validation Accuracy": (
                    round(float(validation_scores[index - 1]), 6)
                    if index <= len(validation_scores)
                    else np.nan
                ),
            }
        )
    history = pd.DataFrame(history_rows)
    history.to_csv(RESULT_DIR / "ann_training_history.csv", index=False)

    importance = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance_df = pd.DataFrame(
        {
            "Feature": x_test.columns,
            "Importance Mean": importance.importances_mean,
            "Importance Standard Deviation": importance.importances_std,
        }
    ).sort_values("Importance Mean", ascending=False)
    importance_df[["Importance Mean", "Importance Standard Deviation"]] = importance_df[
        ["Importance Mean", "Importance Standard Deviation"]
    ].round(6)
    importance_df.to_csv(RESULT_DIR / "ann_permutation_feature_importance.csv", index=False)

    joblib.dump(model, MODEL_DIR / "ann_mlp_model.joblib")
    training_run = {
        "model_type": "scikit-learn MLPClassifier artificial neural network",
        "architecture": "16 inputs -> Dense 32 ReLU -> Dense 16 ReLU -> Sigmoid probability output",
        "overfitting_controls": [
            "L2 regularisation alpha 0.001",
            "20% validation split",
            "early stopping with patience 10",
        ],
        "planned_dropout_note": (
            "The implementation uses L2 regularisation rather than dropout because MLPClassifier "
            "does not expose dropout layers. This controlled change is documented and early stopping is retained."
        ),
        "epochs_completed": int(model.n_iter_),
        "training_seconds": round(training_seconds, 3),
        "decision_threshold": 0.50,
        "random_state": RANDOM_STATE,
        "class_weights": {str(k): float(v) for k, v in class_weights.items()},
    }
    (RESULT_DIR / "ann_training_run.json").write_text(
        json.dumps(training_run, indent=2), encoding="utf-8"
    )

    plt.figure(figsize=(8, 5))
    plt.plot(history["Epoch"], history["Training Loss"], color="#1F4E78", linewidth=2)
    plt.title("ANN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "ann_training_loss.png", dpi=170)
    plt.close()

    if validation_scores:
        plt.figure(figsize=(8, 5))
        plt.plot(history["Epoch"], history["Validation Accuracy"], color="#548235", linewidth=2)
        plt.title("ANN Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(VIS_DIR / "ann_validation_accuracy.png", dpi=170)
        plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay(cm, display_labels=["No churn", "Churn"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("ANN Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(VIS_DIR / "ann_confusion_matrix.png", dpi=170)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    plt.figure(figsize=(7, 5.5))
    plt.plot(fpr, tpr, color="#1F4E78", linewidth=2, label=f"ANN AUC = {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], color="#7F7F7F", linestyle="--", label="Random baseline")
    plt.title("ANN ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "ann_roc_curve.png", dpi=170)
    plt.close()

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, probabilities)
    plt.figure(figsize=(7, 5.5))
    plt.plot(recall_curve, precision_curve, color="#A64B2A", linewidth=2)
    plt.axhline(y_test.mean(), color="#7F7F7F", linestyle="--", label="Churn prevalence")
    plt.title("ANN Precision Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "ann_precision_recall_curve.png", dpi=170)
    plt.close()

    top = importance_df.head(10).sort_values("Importance Mean", ascending=True)
    plt.figure(figsize=(8, 5.5))
    plt.barh(top["Feature"], top["Importance Mean"], color="#2E75B6")
    plt.title("Top ANN Permutation Feature Importance")
    plt.xlabel("Decrease in ROC-AUC when feature is shuffled")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "ann_feature_importance.png", dpi=170)
    plt.close()

    tn, fp, fn, tp = cm.ravel()
    top_features = ", ".join(importance_df.head(5)["Feature"].tolist())
    target_status = metrics_df.loc[
        metrics_df["Target Status"].isin(["Met", "Not met"]),
        ["Metric", "Target Status"],
    ]
    targets_met = int((target_status["Target Status"] == "Met").sum())
    targets_total = len(target_status)

    summary = f"""# ANN Model Training and Evaluation Summary

## Result

The Stage 3 artificial neural network was trained on {len(x_train):,} customer records and evaluated once on the reserved {len(x_test):,}-record testing set. The model completed {model.n_iter_} epochs using early stopping.

## Implementation

- Model: scikit-learn `MLPClassifier`, a feed-forward artificial neural network.
- Architecture: {x_train.shape[1]} inputs, 32-neuron ReLU hidden layer, 16-neuron ReLU hidden layer, and binary probability output.
- Optimizer: Adam with learning rate 0.001.
- Overfitting controls: L2 regularisation at 0.001, 20% validation split, and early stopping with patience 10.
- Class imbalance: balanced sample weights were used during training.
- Decision threshold: 0.50.
- Controlled change: the implemented library does not expose dropout, so L2 regularisation replaced the planned dropout layers. Early stopping and validation monitoring were retained.

## Test Performance

| Metric | Value | Project Target | Status |
| --- | ---: | ---: | --- |
| Accuracy | {metrics['accuracy']:.3f} | {target_map['accuracy']:.2f} | {'Met' if metrics['accuracy'] >= target_map['accuracy'] else 'Not met'} |
| Balanced accuracy | {metrics['balanced_accuracy']:.3f} | Not set | Reported |
| Churn precision | {metrics['precision']:.3f} | Not set | Reported |
| Churn recall | {metrics['recall']:.3f} | {target_map['recall']:.2f} | {'Met' if metrics['recall'] >= target_map['recall'] else 'Not met'} |
| Churn F1-score | {metrics['f1_score']:.3f} | {target_map['f1_score']:.2f} | {'Met' if metrics['f1_score'] >= target_map['f1_score'] else 'Not met'} |
| ROC-AUC | {metrics['roc_auc']:.3f} | {target_map['roc_auc']:.2f} | {'Met' if metrics['roc_auc'] >= target_map['roc_auc'] else 'Not met'} |
| Average precision | {metrics['average_precision']:.3f} | Not set | Reported |

The model met {targets_met} of the {targets_total} measurable project targets.

## Confusion Matrix

- Correctly predicted no churn: {tn}.
- Incorrect churn warnings: {fp}.
- Missed churn cases: {fn}.
- Correctly identified churn cases: {tp}.

## Main Predictive Signals

The five strongest permutation signals were {top_features}. Permutation importance shows which model inputs most affected test-set ROC-AUC when shuffled. It supports interpretation but does not prove that a feature causes churn.

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
"""
    (DOC_DIR / "ann_model_evaluation_summary.md").write_text(summary, encoding="utf-8")

    print("ANN training and evaluation completed.")
    print(f"Epochs: {model.n_iter_}")
    print(metrics_df.to_string(index=False))
    print(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Top features: {top_features}")


if __name__ == "__main__":
    main()
