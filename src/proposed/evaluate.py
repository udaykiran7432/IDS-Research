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

from dqn import D3QN
from data import load_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT / "artifacts" / "phase3" / "d3qn.pt"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "phase3"


def evaluate():
    (
        _,
        _,
        X_test,
        y_test,
        _,
        y_test_original,
    ) = load_data()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model = D3QN(
        state_dim=X_test.shape[1],
        action_dim=2,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    X_tensor = torch.as_tensor(
        X_test,
        dtype=torch.float32,
    )

    with torch.no_grad():
        q_values = model(X_tensor)
        y_pred = torch.argmax(
            q_values,
            dim=1,
        ).numpy()

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

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

    fpr = (
        fp / (fp + tn)
        if (fp + tn)
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else 0.0
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Benign",
            "Ransomware",
        ],
        output_dict=True,
        zero_division=0,
    )

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
        "experiment": "d3qn_evaluation",
        "method": "Dueling Double Deep Q-Network",
        "checkpoint": str(
            CHECKPOINT_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "evaluation_device": "cpu",
        "test_samples": int(len(y_test)),
        "state_dimension": int(X_test.shape[1]),
        "original_test_class_distribution": (
            original_distribution
        ),
        "binary_mapping": {
            "0": "Benign (original S)",
            "1": "Ransomware (original A + SS)",
        },
        "confusion_matrix": {
            "labels": [
                "Benign",
                "Ransomware",
            ],
            "matrix": cm.tolist(),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall_detection_rate": float(
                recall
            ),
            "f1": float(f1),
            "false_positive_rate": float(
                fpr
            ),
            "specificity": float(
                specificity
            ),
        },
        "classification_report": report,
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        REPORT_DIR / "d3qn_evaluation.json"
    )

    with output_path.open("w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print("=== D3QN EVALUATION ===")
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
    print(
        f"Accuracy             : {accuracy:.6f}"
    )
    print(
        f"Precision            : {precision:.6f}"
    )
    print(
        f"Recall / Detection  : {recall:.6f}"
    )
    print(f"F1                   : {f1:.6f}")
    print(
        f"False Positive Rate  : {fpr:.6f}"
    )
    print(
        f"Specificity          : {specificity:.6f}"
    )
    print()
    print("Classification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=[
                "Benign",
                "Ransomware",
            ],
            digits=6,
            zero_division=0,
        )
    )

    print("Saved:", output_path)


if __name__ == "__main__":
    evaluate()
