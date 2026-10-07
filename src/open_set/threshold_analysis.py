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
    / "threshold_analysis_seed_42.json"
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


def main():
    df = pd.read_csv(CALIBRATION_PATH)

    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df["Prediction"].to_numpy(dtype=np.int64)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = D3QN(
        state_dim=8,
        action_dim=2,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    with torch.no_grad():
        q_values = model(
            torch.as_tensor(
                X,
                dtype=torch.float32,
            )
        ).numpy()

    predictions = np.argmax(
        q_values,
        axis=1,
    )

    margins = np.abs(
        q_values[:, 1] - q_values[:, 0]
    )

    correct = predictions == y

    percentiles = [1.0, 2.5, 5.0, 10.0]

    results = []

    print("=== PHASE 7 THRESHOLD SWEEP ===")
    print("Calibration samples:", len(df))
    print("Calibration accuracy:", float(correct.mean()))
    print()

    for percentile in percentiles:
        threshold = float(
            np.percentile(
                margins,
                percentile,
            )
        )

        rejected = margins < threshold

        known_rejection_rate = float(
            rejected.mean()
        )

        correct_rejected = int(
            (rejected & correct).sum()
        )

        incorrect_rejected = int(
            (rejected & ~correct).sum()
        )

        incorrect_total = int(
            (~correct).sum()
        )

        incorrect_capture_rate = float(
            incorrect_rejected
            / max(incorrect_total, 1)
        )

        remaining_known = ~rejected

        remaining_accuracy = float(
            correct[remaining_known].mean()
        ) if remaining_known.any() else 0.0

        result = {
            "percentile": percentile,
            "threshold": threshold,
            "rejected_samples": int(rejected.sum()),
            "known_rejection_rate": known_rejection_rate,
            "correct_predictions_rejected": correct_rejected,
            "incorrect_predictions_rejected": incorrect_rejected,
            "incorrect_capture_rate": incorrect_capture_rate,
            "remaining_known_samples": int(
                remaining_known.sum()
            ),
            "remaining_known_accuracy": remaining_accuracy,
        }

        results.append(result)

        print(
            f"p{percentile:g} | "
            f"threshold={threshold:.9f} | "
            f"rejected={int(rejected.sum())} "
            f"({known_rejection_rate:.4%}) | "
            f"incorrect captured={incorrect_capture_rate:.4%} | "
            f"remaining accuracy={remaining_accuracy:.6f}"
        )

    report = {
        "experiment": "phase7_open_set_threshold_sweep",
        "seed": 42,
        "calibration_samples": int(len(df)),
        "calibration_accuracy": float(correct.mean()),
        "score": "absolute_q_value_margin",
        "unknown_rule": "margin < threshold",
        "candidate_percentiles": percentiles,
        "results": results,
        "locky_test_used_for_calibration": False,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open("w") as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
