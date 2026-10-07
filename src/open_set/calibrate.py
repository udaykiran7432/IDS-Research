import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROPOSED_DIR = PROJECT_ROOT / "src" / "proposed"
if str(PROPOSED_DIR) not in sys.path:
    sys.path.insert(0, str(PROPOSED_DIR))

from dqn import D3QN


CALIBRATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase7_open_set"
    / "calibration.csv"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "phase7"
    / "d3qn_per_open_set_seed_42.pt"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "phase7"
    / "calibration_scores_seed_42.json"
)

FEATURE_COLUMNS = [
    "Time",
    "Protcol",
    "Flag",
    "Family",
    "Clusters",
    "Threats",
    "USD",
    "BTC",
]

TARGET_COLUMN = "Prediction"


def main():
    print("=== PHASE 7 CALIBRATION ===")

    df = pd.read_csv(CALIBRATION_PATH)

    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.int64)

    if not np.isfinite(X).all():
        raise RuntimeError(
            "Calibration features contain non-finite values."
        )

    if not np.isin(y, [0, 1]).all():
        raise RuntimeError(
            "Calibration labels must contain only 0 and 1."
        )

    # Calibration must contain only known families.
    if np.any(df["Family"].to_numpy() == -1):
        raise RuntimeError(
            "Calibration contains unseen-family encoding (-1)."
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = D3QN(
        state_dim=len(FEATURE_COLUMNS),
        action_dim=2,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    with torch.no_grad():
        states = torch.as_tensor(
            X,
            dtype=torch.float32,
        )

        q_values = model(states).cpu().numpy()

    # Greedy D3QN action.
    predicted_action = np.argmax(
        q_values,
        axis=1,
    )

    # Q-value margin:
    # absolute difference between the two action values.
    q_margin = np.abs(
        q_values[:, 1] - q_values[:, 0]
    )

    # Normalized action confidence based on the two Q-values.
    q_min = np.min(q_values, axis=1)
    shifted = q_values - q_min[:, None]
    q_sum = np.sum(shifted, axis=1)

    confidence = np.divide(
        np.max(shifted, axis=1),
        q_sum,
        out=np.full_like(q_sum, 0.5, dtype=np.float64),
        where=q_sum > 0,
    )

    # Prediction correctness.
    correct = predicted_action == y

    print("Calibration samples:", len(df))
    print("Benign:", int((y == 0).sum()))
    print("Ransomware:", int((y == 1).sum()))
    print("Accuracy:", float(correct.mean()))

    print()
    print("=== Q-VALUE MARGIN ===")
    print(
        "min:",
        float(np.min(q_margin)),
    )
    print(
        "median:",
        float(np.median(q_margin)),
    )
    print(
        "p90:",
        float(np.percentile(q_margin, 90)),
    )
    print(
        "p95:",
        float(np.percentile(q_margin, 95)),
    )
    print(
        "p99:",
        float(np.percentile(q_margin, 99)),
    )
    print(
        "max:",
        float(np.max(q_margin)),
    )

    print()
    print("=== CONFIDENCE ===")
    print(
        "min:",
        float(np.min(confidence)),
    )
    print(
        "median:",
        float(np.median(confidence)),
    )
    print(
        "p10:",
        float(np.percentile(confidence, 10)),
    )
    print(
        "p05:",
        float(np.percentile(confidence, 5)),
    )
    print(
        "max:",
        float(np.max(confidence)),
    )

    report = {
        "experiment": "phase7_open_set_calibration",
        "seed": 42,
        "calibration_samples": int(len(df)),
        "benign_samples": int((y == 0).sum()),
        "ransomware_samples": int((y == 1).sum()),
        "baseline_accuracy": float(correct.mean()),
        "q_margin": {
            "min": float(np.min(q_margin)),
            "p01": float(np.percentile(q_margin, 1)),
            "p05": float(np.percentile(q_margin, 5)),
            "p10": float(np.percentile(q_margin, 10)),
            "median": float(np.median(q_margin)),
            "p90": float(np.percentile(q_margin, 90)),
            "p95": float(np.percentile(q_margin, 95)),
            "p99": float(np.percentile(q_margin, 99)),
            "max": float(np.max(q_margin)),
        },
        "confidence": {
            "min": float(np.min(confidence)),
            "p01": float(np.percentile(confidence, 1)),
            "p05": float(np.percentile(confidence, 5)),
            "p10": float(np.percentile(confidence, 10)),
            "median": float(np.median(confidence)),
            "p90": float(np.percentile(confidence, 90)),
            "p95": float(np.percentile(confidence, 95)),
            "p99": float(np.percentile(confidence, 99)),
            "max": float(np.max(confidence)),
        },
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open("w") as f:
        json.dump(report, f, indent=2)

    print()
    print("Calibration report:", REPORT_PATH)


if __name__ == "__main__":
    main()
