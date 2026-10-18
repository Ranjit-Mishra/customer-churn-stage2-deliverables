from pathlib import Path

import json
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def require_file(relative_path: str, minimum_bytes: int = 1) -> Path:
    path = ROOT / relative_path
    if not path.is_file():
        raise AssertionError(f"Missing required file: {relative_path}")
    if path.stat().st_size < minimum_bytes:
        raise AssertionError(f"File is unexpectedly small: {relative_path}")
    return path


def require_text(relative_path: str, phrases: list[str]) -> None:
    text = require_file(relative_path, 100).read_text(encoding="utf-8")
    for phrase in phrases:
        if phrase not in text:
            raise AssertionError(f"Missing '{phrase}' in {relative_path}")
    lowered = text.lower()
    for private_marker in ("reading script", "private video", "recording script"):
        if private_marker in lowered:
            raise AssertionError(
                f"Private meeting material marker found in public repository file: {relative_path}"
            )


def require_image(relative_path: str) -> None:
    path = require_file(relative_path, 5_000)
    with Image.open(path) as image:
        width, height = image.size
        if width < 600 or height < 400:
            raise AssertionError(f"Image resolution is too small: {relative_path}")
        image.verify()


def main() -> None:
    required_files = [
        "Predictive_Modeling/models/ann_mlp_model.joblib",
        "Predictive_Modeling/results/ann_classification_report.csv",
        "Predictive_Modeling/results/ann_confusion_matrix.csv",
        "Predictive_Modeling/results/ann_permutation_feature_importance.csv",
        "Predictive_Modeling/results/ann_test_metrics.csv",
        "Predictive_Modeling/results/ann_test_predictions.csv",
        "Predictive_Modeling/results/ann_training_history.csv",
        "Predictive_Modeling/results/ann_training_run.json",
    ]
    for relative_path in required_files:
        require_file(relative_path, 50)

    metrics = pd.read_csv(ROOT / "Predictive_Modeling/results/ann_test_metrics.csv")
    metric_values = dict(zip(metrics["Metric"], metrics["Value"]))
    expected_metrics = {
        "accuracy": (0.0, 1.0),
        "balanced_accuracy": (0.0, 1.0),
        "precision": (0.0, 1.0),
        "recall": (0.0, 1.0),
        "f1_score": (0.0, 1.0),
        "roc_auc": (0.5, 1.0),
        "average_precision": (0.0, 1.0),
    }
    for name, (lower, upper) in expected_metrics.items():
        if name not in metric_values:
            raise AssertionError(f"Missing metric: {name}")
        value = float(metric_values[name])
        if not lower <= value <= upper:
            raise AssertionError(f"Metric outside expected range: {name}={value}")

    predictions = pd.read_csv(
        ROOT / "Predictive_Modeling/results/ann_test_predictions.csv"
    )
    if len(predictions) != 1_409:
        raise AssertionError(f"Expected 1,409 test predictions, found {len(predictions)}")
    if predictions.isna().any().any():
        raise AssertionError("ANN test predictions contain missing values")

    confusion = pd.read_csv(
        ROOT / "Predictive_Modeling/results/ann_confusion_matrix.csv", index_col=0
    )
    if int(confusion.to_numpy().sum()) != 1_409:
        raise AssertionError("Confusion matrix does not total 1,409 test rows")

    run = json.loads(
        (ROOT / "Predictive_Modeling/results/ann_training_run.json").read_text(
            encoding="utf-8"
        )
    )
    if int(run["epochs_completed"]) < 1:
        raise AssertionError("Training run reports no completed epochs")

    for filename in (
        "ann_confusion_matrix.png",
        "ann_feature_importance.png",
        "ann_precision_recall_curve.png",
        "ann_roc_curve.png",
        "ann_training_loss.png",
        "ann_validation_accuracy.png",
    ):
        require_image(f"Predictive_Modeling/visualisations/{filename}")

    require_text(
        "Predictive_Modeling/docs/ann_model_evaluation_summary.md",
        ["## Test Performance", "ROC-AUC", "## Limitations", "## Decision"],
    )
    require_text(
        "Final_Report/sections/churn_and_retention_factors.md",
        ["## Segment Evidence", "## Predictive Evidence", "## Business Decision"],
    )
    require_text(
        "Final_Report/sections/targeted_retention_recommendations.md",
        ["## Recommended Actions", "## Implementation Sequence", "## Governance"],
    )
    require_text(
        "Final_Report/sections/limitations_and_solutions.md",
        ["Proposed Solution", "Overall Limitation Statement"],
    )

    print("Stage 3 progress verification passed.")
    print("- ANN model, metrics, predictions and six visuals are present and valid.")
    print("- Test predictions and confusion matrix both cover 1,409 records.")
    print("- Business findings, recommendations and limitations are present.")
    print("- No private meeting-script markers were found in public report files.")


if __name__ == "__main__":
    main()
