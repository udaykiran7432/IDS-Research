import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROPOSED_DIR = PROJECT_ROOT / "src" / "proposed"
if str(PROPOSED_DIR) not in sys.path:
    sys.path.insert(0, str(PROPOSED_DIR))

from dqn import D3QN



DATA_DIR = PROJECT_ROOT / "data" / "processed" / "phase7_open_set"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "phase10"
REPORT_DIR = PROJECT_ROOT / "reports" / "phase10"

CALIBRATION_PATH = DATA_DIR / "calibration.csv"
TEST_PATH = DATA_DIR / "test.csv"
METADATA_PATH = DATA_DIR / "test_metadata.csv"


def get_paths(seed):
    checkpoint_path = (
        ARTIFACT_DIR / f"d3qn_per_no_family_seed_{seed}.pt"
    )
    report_path = (
        REPORT_DIR / f"d3qn_per_no_family_evaluation_seed_{seed}.json"
    )
    return checkpoint_path, report_path

FEATURE_COLUMNS = [
    "Time",
    "Protcol",
    "Flag",
    "Clusters",
    "Threats",
    "USD",
    "BTC",
]

TARGET_COLUMN = "Prediction"
TARGET_KNOWN_REJECTION_RATE = 0.05


def load_model(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    saved_features = checkpoint.get("feature_columns")
    if saved_features != FEATURE_COLUMNS:
        raise RuntimeError(
            f"Checkpoint feature mismatch. Expected {FEATURE_COLUMNS}; "
            f"found {saved_features}"
        )

    model = D3QN(state_dim=len(FEATURE_COLUMNS), action_dim=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def get_q_values(model, dataframe):
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(dataframe.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")

    X = dataframe[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = dataframe[TARGET_COLUMN].to_numpy(dtype=np.int64)

    if not np.isfinite(X).all():
        raise RuntimeError("Evaluation features contain non-finite values.")

    with torch.no_grad():
        q_values = model(torch.as_tensor(X, dtype=torch.float32)).numpy()

    return q_values, y


def binary_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    CHECKPOINT_PATH, REPORT_PATH = get_paths(args.seed)

    print("=== PHASE 10: D3QN+PER WITHOUT FAMILY FEATURE ===")

    for path in [
        CHECKPOINT_PATH,
        CALIBRATION_PATH,
        TEST_PATH,
        METADATA_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    calibration = pd.read_csv(CALIBRATION_PATH)
    test = pd.read_csv(TEST_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    if len(test) != len(metadata):
        raise RuntimeError("Test data and metadata row counts differ.")

    if "Family" not in calibration.columns or "Family" not in test.columns:
        raise RuntimeError("Family metadata column is required for integrity checks.")

    if (calibration["Family"].to_numpy() == -1).any():
        raise RuntimeError("Calibration contains unseen-family encoding -1.")

    model, checkpoint = load_model(CHECKPOINT_PATH)

    # Calibration uses only known-family samples and no Family model input.
    q_cal, y_cal = get_q_values(model, calibration)
    margin_cal = np.abs(q_cal[:, 1] - q_cal[:, 0])

    threshold = float(
        np.percentile(margin_cal, 100 * TARGET_KNOWN_REJECTION_RATE)
    )

    # Final evaluation: no threshold selection or tuning on test labels.
    q_test, y_test = get_q_values(model, test)
    margin_test = np.abs(q_test[:, 1] - q_test[:, 0])

    predicted_action = np.argmax(q_test, axis=1)
    predicted_unknown = margin_test < threshold

    families = metadata["Family"].astype(str).to_numpy()
    locky_mask = families == "Locky"
    known_mask = ~locky_mask

    if int(locky_mask.sum()) != 25062:
        raise RuntimeError(
            f"Unexpected Locky count: {int(locky_mask.sum())}"
        )

    if not np.all(test.loc[locky_mask, "Family"].to_numpy() == -1):
        raise RuntimeError(
            "Locky rows are not encoded as Family=-1 in test features."
        )

    if np.any(test.loc[known_mask, "Family"].to_numpy() == -1):
        raise RuntimeError(
            "Known-family test rows contain Family=-1."
        )

    locky_detection = float(predicted_unknown[locky_mask].mean())
    locky_false_accept = float((~predicted_unknown[locky_mask]).mean())
    known_rejection = float(predicted_unknown[known_mask].mean())
    known_acceptance = float((~predicted_unknown[known_mask]).mean())

    retained_known = known_mask & ~predicted_unknown

    report = {
        "experiment": "phase10_d3qn_per_no_family_open_set",
        "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
        "seed": int(checkpoint["seed"]),
        "feature_columns": FEATURE_COLUMNS,
        "family_feature_excluded": True,
        "calibration_samples": int(len(calibration)),
        "test_samples": int(len(test)),
        "known_test_samples": int(known_mask.sum()),
        "locky_test_samples": int(locky_mask.sum()),
        "threshold_selection": (
            "5th percentile of absolute Q-value margins from known-family "
            "calibration data; test data not used for threshold selection."
        ),
        "target_known_rejection_rate": TARGET_KNOWN_REJECTION_RATE,
        "calibration_threshold": threshold,
        "observed_calibration_rejection_rate": float(
            np.mean(margin_cal < threshold)
        ),
        "open_set_metrics": {
            "locky_unknown_detection_rate": locky_detection,
            "locky_false_accept_rate": locky_false_accept,
            "known_rejection_rate": known_rejection,
            "known_acceptance_rate": known_acceptance,
        },
        "known_closed_set_metrics_before_rejection": binary_metrics(
            y_test[known_mask],
            predicted_action[known_mask],
        ),
        "known_retained_metrics_after_rejection": (
            binary_metrics(
                y_test[retained_known],
                predicted_action[retained_known],
            )
            if retained_known.any()
            else None
        ),
        "notes": [
            "Family is excluded from the seven model input features.",
            "Family metadata is used only for split integrity and evaluation.",
            "The short smoke-test checkpoint is not a research result.",
            "The threshold is selected from known-family calibration margins.",
            "Locky test labels are not used for threshold selection.",
        ],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("Checkpoint seed:", checkpoint["seed"])
    print("Features:", FEATURE_COLUMNS)
    print("Calibration threshold:", threshold)
    print("Observed calibration rejection rate:",
          report["observed_calibration_rejection_rate"])
    print("Known-family rejection rate:", known_rejection)
    print("Locky unknown-detection rate:", locky_detection)
    print("Locky false-accept rate:", locky_false_accept)
    print("Report:", REPORT_PATH)


if __name__ == "__main__":
    main()
