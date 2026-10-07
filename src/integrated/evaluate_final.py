from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

import sys




PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROPOSED_DIR = PROJECT_ROOT / "src" / "proposed"
if str(PROPOSED_DIR) not in sys.path:
    sys.path.insert(0, str(PROPOSED_DIR))

from agent import D3QN


TEST_PATH = PROJECT_ROOT / "data/processed/phase8_integrated/test.csv"
META_PATH = PROJECT_ROOT / "data/processed/phase7_open_set/test_metadata.csv"
CHECKPOINT = PROJECT_ROOT / "artifacts/phase8/d3qn_per_vae_seed_42.pt"

REPORT_PATH = (
    PROJECT_ROOT
    / "reports/phase8/final_e6_open_set_evaluation_seed_42.json"
)

FEATURE_COLUMNS = [
    "latent_0",
    "latent_1",
    "latent_2",
    "latent_3",
    "reconstruction_error",
]

THRESHOLD = 0.23906713426113124


def binary_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "confusion_matrix": cm.tolist(),
    }


def reconstruction_stats(values):
    values = np.asarray(values, dtype=np.float64)

    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def main():
    print("=== PHASE 8 FINAL E6 HELD-OUT TEST ===")
    print("Checkpoint:", CHECKPOINT)
    print("Test:", TEST_PATH)
    print("Metadata:", META_PATH)
    print("Frozen VAE threshold:", THRESHOLD)
    print()

    assert TEST_PATH.exists()
    assert META_PATH.exists()
    assert CHECKPOINT.exists()

    test_df = pd.read_csv(TEST_PATH)
    meta_df = pd.read_csv(META_PATH)

    assert list(test_df[FEATURE_COLUMNS].columns) == FEATURE_COLUMNS
    assert len(test_df) == len(meta_df)
    assert len(test_df) == 62257
    assert test_df["Prediction"].isin([0, 1]).all()
    assert np.isfinite(
        test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    ).all()

    # Metadata is used ONLY for evaluation/identification.
    family_values = meta_df["Family"].astype(str).to_numpy()

    locky_mask = family_values == "Locky"
    known_mask = ~locky_mask

    assert int(locky_mask.sum()) == 25062
    assert int(known_mask.sum()) == 37195

    X = test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = test_df["Prediction"].to_numpy(dtype=np.int64)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    state_dim = int(checkpoint["state_dim"])
    assert state_dim == 5

    model = D3QN(
        state_dim=5,
        action_dim=2,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    with torch.no_grad():
        states = torch.tensor(X, dtype=torch.float32)
        q_values = model(states).cpu().numpy()

    dqn_predictions = np.argmax(q_values, axis=1).astype(np.int64)

    reconstruction_error = X[:, 4]

    unknown_mask = reconstruction_error >= THRESHOLD
    accepted_known_mask = ~unknown_mask

    # Final open-set decision:
    # high reconstruction error -> UNKNOWN
    # otherwise use D3QN class prediction.
    final_predictions = dqn_predictions.copy()
    final_predictions[unknown_mask] = -1

    # ------------------------------------------------------------------
    # Basic counts
    # ------------------------------------------------------------------

    locky_unknown = int(np.sum(unknown_mask & locky_mask))
    locky_known = int(np.sum(accepted_known_mask & locky_mask))

    known_unknown = int(np.sum(unknown_mask & known_mask))
    known_accepted = int(np.sum(accepted_known_mask & known_mask))

    unknown_detection_rate = (
        locky_unknown / int(locky_mask.sum())
    )

    unknown_false_accept_rate = (
        locky_known / int(locky_mask.sum())
    )

    known_rejection_rate = (
        known_unknown / int(known_mask.sum())
    )

    known_acceptance_rate = (
        known_accepted / int(known_mask.sum())
    )

    # ------------------------------------------------------------------
    # Closed-set performance on known families
    # ------------------------------------------------------------------

    known_closed_metrics = binary_metrics(
        y[known_mask],
        dqn_predictions[known_mask],
    )

    # ------------------------------------------------------------------
    # Performance on known samples retained by open-set mechanism
    # ------------------------------------------------------------------

    retained_known_mask = known_mask & accepted_known_mask

    known_retained_metrics = binary_metrics(
        y[retained_known_mask],
        dqn_predictions[retained_known_mask],
    )

    # ------------------------------------------------------------------
    # Locky accepted as known: diagnostic
    # ------------------------------------------------------------------

    locky_accepted_predictions = dqn_predictions[
        locky_mask & accepted_known_mask
    ]

    locky_accepted_benign = int(
        np.sum(locky_accepted_predictions == 0)
    )

    locky_accepted_ransomware = int(
        np.sum(locky_accepted_predictions == 1)
    )

    # ------------------------------------------------------------------
    # Reconstruction-error separation
    # ------------------------------------------------------------------

    known_errors = reconstruction_error[known_mask]
    locky_errors = reconstruction_error[locky_mask]

    # Higher reconstruction error = more likely unseen.
    y_unknown = locky_mask.astype(np.int64)

    try:
        reconstruction_auc = float(
            roc_auc_score(
                y_unknown,
                reconstruction_error,
            )
        )
    except ValueError:
        reconstruction_auc = None

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------

    print("Total samples:", len(test_df))
    print("Known samples:", int(known_mask.sum()))
    print("Unseen Locky samples:", int(locky_mask.sum()))
    print()

    print("=== OPEN-SET RESULT ===")
    print("Frozen threshold:", THRESHOLD)
    print("Locky -> UNKNOWN:", locky_unknown)
    print("Locky -> KNOWN:", locky_known)
    print(
        "Unknown detection rate:",
        f"{unknown_detection_rate:.6f}",
    )
    print(
        "Unknown false-accept rate:",
        f"{unknown_false_accept_rate:.6f}",
    )
    print(
        "Known -> UNKNOWN:",
        known_unknown,
    )
    print(
        "Known rejection rate:",
        f"{known_rejection_rate:.6f}",
    )
    print(
        "Known acceptance rate:",
        f"{known_acceptance_rate:.6f}",
    )
    print()

    print("=== KNOWN TRAFFIC: CLOSED SET ===")
    print(
        json.dumps(
            known_closed_metrics,
            indent=2,
        )
    )
    print()

    print("=== KNOWN TRAFFIC: RETAINED AFTER OPEN-SET FILTER ===")
    print(
        json.dumps(
            known_retained_metrics,
            indent=2,
        )
    )
    print()

    print("=== LOCKY ACCEPTED AS KNOWN ===")
    print("Benign:", locky_accepted_benign)
    print("Ransomware:", locky_accepted_ransomware)
    print()

    print("=== RECONSTRUCTION ERROR ===")
    print("Known:")
    print(json.dumps(reconstruction_stats(known_errors), indent=2))
    print()
    print("Locky:")
    print(json.dumps(reconstruction_stats(locky_errors), indent=2))
    print()
    print("Reconstruction-error ROC-AUC:", reconstruction_auc)
    print()

    report = {
        "experiment": "E6_D3QN_PER_VAE_OPEN_SET",
        "seed": 42,
        "checkpoint": str(CHECKPOINT),
        "test_samples": int(len(test_df)),
        "known_samples": int(known_mask.sum()),
        "unseen_family": "Locky",
        "unseen_samples": int(locky_mask.sum()),
        "feature_columns": FEATURE_COLUMNS,
        "state_dimension": 5,
        "open_set_signal": "vae_reconstruction_error",
        "threshold": THRESHOLD,
        "threshold_source": (
            "95th_percentile_of_known_calibration_reconstruction_error"
        ),
        "test_threshold_tuning": False,
        "open_set": {
            "locky_unknown": locky_unknown,
            "locky_known": locky_known,
            "unknown_detection_rate": unknown_detection_rate,
            "unknown_false_accept_rate": unknown_false_accept_rate,
            "known_unknown": known_unknown,
            "known_accepted": known_accepted,
            "known_rejection_rate": known_rejection_rate,
            "known_acceptance_rate": known_acceptance_rate,
        },
        "known_closed_set_metrics": known_closed_metrics,
        "known_retained_metrics": known_retained_metrics,
        "locky_accepted_as_known": {
            "benign": locky_accepted_benign,
            "ransomware": locky_accepted_ransomware,
        },
        "reconstruction_error": {
            "known": reconstruction_stats(known_errors),
            "locky": reconstruction_stats(locky_errors),
            "roc_auc": reconstruction_auc,
        },
        "test_data_used": True,
        "locky_data_used": True,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
