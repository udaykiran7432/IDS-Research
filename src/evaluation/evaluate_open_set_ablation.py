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
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TEST_PATH = (
    PROJECT_ROOT
    / "data/processed/phase8_integrated/test.csv"
)

META_PATH = (
    PROJECT_ROOT
    / "data/processed/phase7_open_set/test_metadata.csv"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts/phase8/d3qn_per_vae_seed_42.pt"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports/phase10/open_set_ablation_seed_42.json"
)

FEATURE_COLUMNS = [
    "latent_0",
    "latent_1",
    "latent_2",
    "latent_3",
    "reconstruction_error",
]

THRESHOLD = 0.23906713426113124


def metrics(y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
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
        if (fp + tn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "specificity": float(specificity),
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------
# Load authoritative held-out test
# ---------------------------------------------------------
test_df = pd.read_csv(TEST_PATH)
meta_df = pd.read_csv(META_PATH)

assert len(test_df) == 62257
assert len(meta_df) == 62257

X = test_df[
    FEATURE_COLUMNS
].to_numpy(dtype=np.float32)

y = test_df[
    "Prediction"
].to_numpy(dtype=np.int64)

family = (
    meta_df["Family"]
    .astype(str)
    .to_numpy()
)

locky_mask = family == "Locky"
known_mask = ~locky_mask

assert int(locky_mask.sum()) == 25062
assert int(known_mask.sum()) == 37195

assert np.isfinite(X).all()
assert set(np.unique(y)).issubset({0, 1})


# ---------------------------------------------------------
# Load frozen E6 D3QN+PER model
# ---------------------------------------------------------
checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

assert checkpoint["state_dim"] == 5

from proposed.dqn import D3QN

model = D3QN(
    state_dim=5,
    action_dim=2,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ---------------------------------------------------------
# D3QN predictions
# ---------------------------------------------------------
with torch.no_grad():
    states = torch.tensor(
        X,
        dtype=torch.float32,
    )

    q_values = model(states).numpy()

dqn_predictions = np.argmax(
    q_values,
    axis=1,
).astype(np.int64)


# ---------------------------------------------------------
# Open-set signal
# ---------------------------------------------------------
reconstruction_error = X[:, 4]

unknown_mask = (
    reconstruction_error >= THRESHOLD
)

accepted_mask = ~unknown_mask


# =========================================================
# A. CLOSED SET
# =========================================================
# Every sample is forced into benign/ransomware.
# The open-set mechanism does not exist.

known_closed_metrics = metrics(
    y[known_mask],
    dqn_predictions[known_mask],
)

locky_closed_predictions = (
    dqn_predictions[locky_mask]
)

locky_closed_distribution = {
    "benign": int(
        np.sum(locky_closed_predictions == 0)
    ),
    "ransomware": int(
        np.sum(locky_closed_predictions == 1)
    ),
}


# =========================================================
# B. OPEN SET
# =========================================================
final_predictions = dqn_predictions.copy()

final_predictions[unknown_mask] = -1

known_unknown = int(
    np.sum(unknown_mask & known_mask)
)

known_accepted = int(
    np.sum(accepted_mask & known_mask)
)

locky_unknown = int(
    np.sum(unknown_mask & locky_mask)
)

locky_accepted = int(
    np.sum(accepted_mask & locky_mask)
)

known_rejection_rate = (
    known_unknown / int(known_mask.sum())
)

locky_unknown_rate = (
    locky_unknown / int(locky_mask.sum())
)

locky_false_accept_rate = (
    locky_accepted / int(locky_mask.sum())
)

retained_known_mask = (
    known_mask & accepted_mask
)

known_retained_metrics = metrics(
    y[retained_known_mask],
    dqn_predictions[retained_known_mask],
)


# ---------------------------------------------------------
# Open-set benefit relative to closed-set
# ---------------------------------------------------------
# Closed-set has zero unknown detection by definition.
#
# Open-set additionally identifies unseen Locky.
#
# Known classification is evaluated separately so that
# UNKNOWN is not incorrectly treated as a classification
# class.

report = {
    "experiment": "Phase10.4_open_set_ablation",
    "seed": 42,
    "checkpoint": str(
        CHECKPOINT.relative_to(PROJECT_ROOT)
    ),
    "test_samples": int(len(test_df)),
    "known_samples": int(known_mask.sum()),
    "unseen_locky_samples": int(locky_mask.sum()),
    "threshold": THRESHOLD,
    "open_set_signal": "vae_reconstruction_error",

    "closed_set": {
        "known_metrics": known_closed_metrics,
        "locky_unknown_detection_rate": 0.0,
        "locky_false_accept_rate": 1.0,
        "locky_predictions": locky_closed_distribution,
    },

    "open_set": {
        "known_unknown": known_unknown,
        "known_accepted": known_accepted,
        "known_rejection_rate": float(
            known_rejection_rate
        ),
        "known_acceptance_rate": float(
            known_accepted / int(known_mask.sum())
        ),
        "locky_unknown": locky_unknown,
        "locky_accepted": locky_accepted,
        "locky_unknown_detection_rate": float(
            locky_unknown_rate
        ),
        "locky_false_accept_rate": float(
            locky_false_accept_rate
        ),
        "known_retained_metrics": known_retained_metrics,
    },

    "ablation_delta": {
        "locky_unknown_detection_rate": float(
            locky_unknown_rate
        ),
        "locky_false_accept_rate_reduction": float(
            1.0 - locky_false_accept_rate
        ),
    },
}


REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)


# ---------------------------------------------------------
# Console
# ---------------------------------------------------------
print("=" * 70)
print("PHASE 10.4 — OPEN-SET MECHANISM ABLATION")
print("=" * 70)

print()
print("Test samples:", len(test_df))
print("Known samples:", int(known_mask.sum()))
print("Locky samples:", int(locky_mask.sum()))
print("Frozen threshold:", THRESHOLD)

print()
print("=== CLOSED SET ===")
print("Known traffic:")
print(json.dumps(
    known_closed_metrics,
    indent=2,
))

print()
print("Locky forced into known classes:")
print(locky_closed_distribution)

print("Locky UNKNOWN detection: 0.000000")
print("Locky false acceptance: 1.000000")

print()
print("=== OPEN SET ===")
print("Known -> UNKNOWN:", known_unknown)
print("Known rejection rate:", known_rejection_rate)
print("Known accepted:", known_accepted)

print()
print("Locky -> UNKNOWN:", locky_unknown)
print("Locky -> KNOWN:", locky_accepted)
print(
    "Locky UNKNOWN detection rate:",
    locky_unknown_rate,
)
print(
    "Locky false-accept rate:",
    locky_false_accept_rate,
)

print()
print("Known retained metrics:")
print(json.dumps(
    known_retained_metrics,
    indent=2,
))

print()
print("Report saved:", REPORT_PATH)
print("=" * 70)
