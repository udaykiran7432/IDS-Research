import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from dqn import DQN
from data import load_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "phase2" / "baseline_dqn.pt"
REPORT_DIR = PROJECT_ROOT / "reports" / "phase2"


def evaluate(
    checkpoint_path=None,
    report_dir=None,
):
    checkpoint_path = (
    Path(checkpoint_path).resolve()
    if checkpoint_path
    else CHECKPOINT_PATH
)
    report_dir = (
        Path(report_dir)
        if report_dir
        else REPORT_DIR
    )
    (
    _,
    _,
    X_test,
    y_test,
    _,
    y_test_original,
    ) = load_data()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model = DQN(
        state_dim=X_test.shape[1],
        action_dim=2,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X_tensor = torch.as_tensor(
        X_test,
        dtype=torch.float32,
    )

    with torch.no_grad():
        q_values = model(X_tensor)
        y_pred = torch.argmax(q_values, dim=1).numpy()

    # Binary metrics.
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Ransomware"],
        output_dict=True,
        zero_division=0,
    )

    # Keep original 3-class test distribution for traceability.
    original_distribution = {
        str(int(label)): int(count)
        for label, count in zip(
            *np.unique(
                y_test_original,
                return_counts=True,
            )
        )
    }

    results = {
        "experiment": "baseline_dqn_evaluation",
        "checkpoint": str(
            checkpoint_path.relative_to(PROJECT_ROOT)
        ),
        "evaluation_device": "cpu",
        "test_samples": int(len(y_test)),
        "state_dimension": int(X_test.shape[1]),
        "original_test_class_distribution": original_distribution,
        "binary_mapping": {
            "0": "Benign (original S)",
            "1": "Ransomware (original A + SS)",
        },
        "confusion_matrix": {
            "labels": ["Benign", "Ransomware"],
            "matrix": cm.tolist(),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall_detection_rate": float(recall),
            "f1": float(f1),
            "false_positive_rate": float(fpr),
            "specificity": float(specificity),
        },
        "classification_report": report,
    }

    report_dir.mkdir(parents=True, exist_ok=True)

    output_path = report_dir / "baseline_evaluation.json"

    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print("=== BASELINE EVALUATION ===")
    print(f"Test samples : {len(y_test)}")
    print()
    print("Confusion Matrix")
    print(cm)
    print()
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")
    print()
    print(f"Accuracy             : {accuracy:.6f}")
    print(f"Precision            : {precision:.6f}")
    print(f"Recall / Detection  : {recall:.6f}")
    print(f"F1                   : {f1:.6f}")
    print(f"False Positive Rate  : {fpr:.6f}")
    print(f"Specificity          : {specificity:.6f}")
    print()
    print("Classification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Benign", "Ransomware"],
            digits=6,
            zero_division=0,
        )
    )
    print("Saved:", output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        report_dir=args.report_dir,
    )
