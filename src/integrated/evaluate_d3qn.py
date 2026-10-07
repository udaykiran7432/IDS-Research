from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROPOSED_DIR = PROJECT_ROOT / "src" / "proposed"
if str(PROPOSED_DIR) not in sys.path:
    sys.path.insert(0, str(PROPOSED_DIR))

from dqn import D3QN


CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "phase8"
    / "d3qn_per_vae_seed_42.pt"
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase8_integrated"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "phase8"
    / "d3qn_pre_test_evaluation_seed_42.json"
)

FEATURE_COLUMNS = [
    "latent_0",
    "latent_1",
    "latent_2",
    "latent_3",
    "reconstruction_error",
]

FROZEN_VAE_THRESHOLD = 0.23906713426113124


def binary_metrics(y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
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
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Benign",
            "Ransomware",
        ],
        output_dict=True,
        zero_division=0,
    )

    return {
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
            "false_positive_rate": float(fpr),
            "specificity": float(
                specificity
            ),
        },
        "classification_report": report,
    }


def evaluate_split(
    model,
    split_name: str,
):
    path = DATA_DIR / f"{split_name}.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    expected_columns = (
        FEATURE_COLUMNS + ["Prediction"]
    )

    if list(df.columns) != expected_columns:
        raise RuntimeError(
            f"Unexpected columns in {path}: "
            f"{list(df.columns)}"
        )

    X = df[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float32)

    y = df[
        "Prediction"
    ].to_numpy(dtype=np.int64)

    if not np.isfinite(X).all():
        raise RuntimeError(
            f"{split_name} contains non-finite "
            "representation values."
        )

    if not np.isin(y, [0, 1]).all():
        raise RuntimeError(
            f"{split_name} contains non-binary targets."
        )

    with torch.no_grad():
        q_values = model(
            torch.as_tensor(
                X,
                dtype=torch.float32,
            )
        ).numpy()

    predicted_action = np.argmax(
        q_values,
        axis=1,
    )

    metrics = binary_metrics(
        y,
        predicted_action,
    )

    reconstruction_error = X[:, 4]

    rejected_by_vae = (
        reconstruction_error
        >= FROZEN_VAE_THRESHOLD
    )

    accepted_by_vae = ~rejected_by_vae

    result = {
        "samples": int(len(df)),
        "binary_distribution": {
            "benign": int((y == 0).sum()),
            "ransomware": int((y == 1).sum()),
        },
        "d3qn_predictions": {
            "benign": int(
                (predicted_action == 0).sum()
            ),
            "ransomware": int(
                (predicted_action == 1).sum()
            ),
        },
        "binary_metrics": metrics,
        "vae_reconstruction_error": {
            "min": float(
                reconstruction_error.min()
            ),
            "median": float(
                np.median(reconstruction_error)
            ),
            "p95": float(
                np.percentile(
                    reconstruction_error,
                    95,
                )
            ),
            "max": float(
                reconstruction_error.max()
            ),
        },
        "frozen_vae_threshold": (
            FROZEN_VAE_THRESHOLD
        ),
        "vae_threshold_behavior": {
            "rejected_as_unknown": int(
                rejected_by_vae.sum()
            ),
            "accepted_as_known": int(
                accepted_by_vae.sum()
            ),
            "rejection_rate": float(
                rejected_by_vae.mean()
            ),
        },
    }

    return result


def main():
    print(
        "=== PHASE 8 D3QN PRE-TEST EVALUATION ==="
    )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("state_dim") != 5:
        raise RuntimeError(
            "Checkpoint state_dim is not 5."
        )

    model = D3QN(
        state_dim=5,
        action_dim=2,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )
    print(
        "State dimension:",
        checkpoint["state_dim"],
    )
    print(
        "Frozen VAE threshold:",
        FROZEN_VAE_THRESHOLD,
    )

    # IMPORTANT:
    # Only train and calibration are evaluated here.
    # The held-out test split is deliberately NOT loaded.
    results = {
        "experiment": (
            "Phase 8 D3QN + PER + VAE "
            "pre-test evaluation"
        ),
        "checkpoint": str(
            CHECKPOINT_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "evaluation_device": "cpu",
        "state_dimension": 5,
        "feature_columns": FEATURE_COLUMNS,
        "frozen_vae_threshold": (
            FROZEN_VAE_THRESHOLD
        ),
        "test_data_used": False,
        "locky_data_used": False,
        "splits_evaluated": [
            "train",
            "calibration",
        ],
        "splits": {},
    }

    for split_name in [
        "train",
        "calibration",
    ]:
        print()
        print(
            f"=== {split_name.upper()} ==="
        )

        result = evaluate_split(
            model,
            split_name,
        )

        results["splits"][split_name] = result

        metrics = result["binary_metrics"][
            "metrics"
        ]

        print(
            "Samples:",
            result["samples"],
        )

        print(
            "Confusion matrix:"
        )

        print(
            result["binary_metrics"][
                "confusion_matrix"
            ]["matrix"]
        )

        print(
            f"Accuracy            : "
            f"{metrics['accuracy']:.6f}"
        )

        print(
            f"Precision           : "
            f"{metrics['precision']:.6f}"
        )

        print(
            f"Recall / Detection : "
            f"{metrics['recall_detection_rate']:.6f}"
        )

        print(
            f"F1                  : "
            f"{metrics['f1']:.6f}"
        )

        print(
            f"False Positive Rate : "
            f"{metrics['false_positive_rate']:.6f}"
        )

        print(
            f"Specificity         : "
            f"{metrics['specificity']:.6f}"
        )

        print(
            "VAE threshold rejects:",
            result[
                "vae_threshold_behavior"
            ]["rejected_as_unknown"],
        )

        print(
            "VAE threshold rejection rate:",
            f"{result['vae_threshold_behavior']['rejection_rate']:.6f}",
        )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open("w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print(
        "Test data used: False"
    )
    print(
        "Locky data used: False"
    )
    print()
    print(
        "Saved:",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()
