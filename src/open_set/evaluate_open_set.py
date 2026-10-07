import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sys

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROPOSED_DIR = PROJECT_ROOT / "src" / "proposed"
if str(PROPOSED_DIR) not in sys.path:
    sys.path.insert(0, str(PROPOSED_DIR))

from dqn import D3QN


TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase7_open_set"
    / "test.csv"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase7_open_set"
    / "test_metadata.csv"
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
    / "open_set_evaluation_seed_42.json"
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

THRESHOLD = 0.3517799377441406


def binary_metrics(y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "fpr": float(
            fp / max(fp + tn, 1)
        ),
        "specificity": float(
            tn / max(tn + fp, 1)
        ),
        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
    }


def main():
    print("=== PHASE 7 OPEN-SET EVALUATION ===")

    test_df = pd.read_csv(TEST_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    if len(test_df) != len(metadata):
        raise RuntimeError(
            "Test and metadata row counts do not match."
        )

    X = test_df[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32
    )

    y = test_df["Prediction"].to_numpy(
        dtype=np.int64
    )

    families = metadata["Family"].astype(str).to_numpy()

    locky_mask = families == "Locky"
    known_mask = ~locky_mask

    if int(locky_mask.sum()) != 25062:
        raise RuntimeError(
            "Unexpected Locky test count."
        )

    if np.any(
        test_df.loc[known_mask, "Family"].to_numpy() == -1
    ):
        raise RuntimeError(
            "Known test samples contain unseen-family encoding."
        )

    if not np.all(
        test_df.loc[locky_mask, "Family"].to_numpy() == -1
    ):
        raise RuntimeError(
            "Locky samples are not encoded as unseen family."
        )

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

    predicted_action = np.argmax(
        q_values,
        axis=1,
    )

    q_margin = np.abs(
        q_values[:, 1] - q_values[:, 0]
    )

    unknown_predicted = q_margin < THRESHOLD
    known_predicted = ~unknown_predicted

    # ---------------------------------------------------------
    # Known-vs-unknown detection
    # ---------------------------------------------------------
    actual_unknown = locky_mask
    actual_known = known_mask

    open_set_predicted_unknown = unknown_predicted

    unknown_detection_rate = float(
        (
            open_set_predicted_unknown
            & actual_unknown
        ).sum()
        / max(actual_unknown.sum(), 1)
    )

    unknown_false_accept_rate = float(
        (
            known_predicted
            & actual_unknown
        ).sum()
        / max(actual_unknown.sum(), 1)
    )

    known_rejection_rate = float(
        (
            unknown_predicted
            & actual_known
        ).sum()
        / max(actual_known.sum(), 1)
    )

    known_acceptance_rate = float(
        (
            known_predicted
            & actual_known
        ).sum()
        / max(actual_known.sum(), 1)
    )

    # ---------------------------------------------------------
    # Binary classification on all samples
    # ---------------------------------------------------------
    # UNKNOWN samples are excluded from ordinary binary
    # classification metrics because UNKNOWN is a separate
    # open-set decision, not a binary attack/benign class.
    known_retained = actual_known & known_predicted

    known_binary_metrics = None

    if known_retained.any():
        known_binary_metrics = binary_metrics(
            y[known_retained],
            predicted_action[known_retained],
        )

    # Binary performance on all known samples before
    # open-set rejection.
    known_closed_set_metrics = binary_metrics(
        y[actual_known],
        predicted_action[actual_known],
    )

    # ---------------------------------------------------------
    # Family-level Locky results
    # ---------------------------------------------------------
    locky_y = y[locky_mask]
    locky_actions = predicted_action[locky_mask]
    locky_margins = q_margin[locky_mask]

    locky_result = {
        "samples": int(locky_mask.sum()),
        "binary_benign": int((locky_y == 0).sum()),
        "binary_ransomware": int((locky_y == 1).sum()),
        "unknown_detected": int(
            unknown_predicted[locky_mask].sum()
        ),
        "accepted_as_known": int(
            known_predicted[locky_mask].sum()
        ),
        "unknown_detection_rate": unknown_detection_rate,
        "false_accept_rate": unknown_false_accept_rate,
        "q_margin": {
            "min": float(np.min(locky_margins)),
            "p05": float(np.percentile(locky_margins, 5)),
            "p25": float(np.percentile(locky_margins, 25)),
            "median": float(np.median(locky_margins)),
            "p75": float(np.percentile(locky_margins, 75)),
            "p95": float(np.percentile(locky_margins, 95)),
            "max": float(np.max(locky_margins)),
        },
        "accepted_known_binary_prediction": {
            "benign": int(
                (
                    known_predicted[locky_mask]
                    & (locky_actions == 0)
                ).sum()
            ),
            "ransomware": int(
                (
                    known_predicted[locky_mask]
                    & (locky_actions == 1)
                ).sum()
            ),
        },
    }

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------
    print()
    print("Frozen threshold:", THRESHOLD)

    print()
    print("=== TEST POPULATION ===")
    print("Total:", len(test_df))
    print("Known:", int(actual_known.sum()))
    print("Unseen Locky:", int(actual_unknown.sum()))

    print()
    print("=== OPEN-SET DETECTION ===")
    print(
        "Locky detected UNKNOWN:",
        locky_result["unknown_detected"],
    )
    print(
        "Locky accepted as KNOWN:",
        locky_result["accepted_as_known"],
    )
    print(
        "Locky Unknown Detection Rate:",
        f"{unknown_detection_rate:.6f}",
    )
    print(
        "Locky False Accept Rate:",
        f"{unknown_false_accept_rate:.6f}",
    )

    print()
    print("Known rejected as UNKNOWN:", int(
        (
            unknown_predicted
            & actual_known
        ).sum()
    ))

    print(
        "Known accepted as KNOWN:",
        int(
            (
                known_predicted
                & actual_known
            ).sum()
        ),
    )

    print(
        "Known rejection rate:",
        f"{known_rejection_rate:.6f}",
    )

    print()
    print("=== KNOWN CLOSED-SET METRICS ===")
    print(json.dumps(
        known_closed_set_metrics,
        indent=2,
    ))

    print()
    print("=== KNOWN RETAINED AS KNOWN ===")
    print(json.dumps(
        known_binary_metrics,
        indent=2,
    ))

    print()
    print("=== LOCKY Q-MARGIN ===")
    print(json.dumps(
        locky_result["q_margin"],
        indent=2,
    ))

    # ---------------------------------------------------------
    # Save report
    # ---------------------------------------------------------
    report = {
        "experiment": "phase7_family_held_out_open_set",
        "seed": 42,
        "unseen_family": "Locky",
        "threshold": THRESHOLD,
        "threshold_selection": (
            "5th percentile of known-family calibration "
            "Q-value margin; Locky test not used."
        ),
        "test_population": {
            "total": int(len(test_df)),
            "known": int(actual_known.sum()),
            "unseen_locky": int(actual_unknown.sum()),
        },
        "open_set_detection": {
            "unknown_detection_rate": unknown_detection_rate,
            "unknown_false_accept_rate": unknown_false_accept_rate,
            "known_rejection_rate": known_rejection_rate,
            "known_acceptance_rate": known_acceptance_rate,
        },
        "known_closed_set_metrics": known_closed_set_metrics,
        "known_retained_as_known_metrics": known_binary_metrics,
        "locky": locky_result,
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
