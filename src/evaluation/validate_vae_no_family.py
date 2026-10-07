import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.vae.model import VAE


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
SEED = 42

CHECKPOINT = PROJECT_ROOT / "artifacts/phase10/vae_no_family_seed_42.pt"

TRAIN_PATH = PROJECT_ROOT / "data/processed/phase7_open_set/train.csv"
CAL_PATH = PROJECT_ROOT / "data/processed/phase7_open_set/calibration.csv"
TEST_PATH = PROJECT_ROOT / "data/processed/phase7_open_set/test.csv"
METADATA_PATH = PROJECT_ROOT / "data/processed/phase7_open_set/test_metadata.csv"

REPORT_PATH = PROJECT_ROOT / "reports/phase10/e4b_vae_validation_seed_42.json"

FEATURE_COLUMNS = [
    "Time",
    "Protcol",
    "Flag",
    "Clusters",
    "Threats",
    "USD",
    "BTC",
]

FAMILY_COLUMN = "Family"

LATENT_DIM = 4


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)
print("Checkpoint:", CHECKPOINT)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def load_features(path):
    df = pd.read_csv(path)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features in {path}: {missing}")

    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    return df, X


def reconstruction_errors(model, X):
    model.eval()

    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32, device=DEVICE)

        mu, logvar = model.encode(x)

        # Deterministic representation:
        # use latent mean rather than stochastic sampling.
        reconstruction = model.decode(mu)

        errors = torch.mean((reconstruction - x) ** 2, dim=1)

    return errors.cpu().numpy()


def summarize(name, errors):
    errors = np.asarray(errors)

    return {
        "name": name,
        "count": int(len(errors)),
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "median": float(np.percentile(errors, 50)),
        "p95": float(np.percentile(errors, 95)),
        "p99": float(np.percentile(errors, 99)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
    }


# ---------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------
checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

assert checkpoint["input_dim"] == len(FEATURE_COLUMNS)
assert checkpoint["latent_dim"] == LATENT_DIM
assert FAMILY_COLUMN not in checkpoint["feature_columns"]

model = VAE(
    input_dim=checkpoint["input_dim"],
    hidden_dim=checkpoint["hidden_dim"],
    latent_dim=checkpoint["latent_dim"],
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print("Features:", FEATURE_COLUMNS)


# ---------------------------------------------------------
# Load Phase 7 family-held-out data
# ---------------------------------------------------------
train_df, X_train = load_features(TRAIN_PATH)
cal_df, X_cal = load_features(CAL_PATH)
test_df, X_test = load_features(TEST_PATH)

metadata = pd.read_csv(METADATA_PATH)

if len(test_df) != len(metadata):
    raise ValueError(
        f"Test/metadata size mismatch: {len(test_df)} vs {len(metadata)}"
    )

print()
print("Shapes:")
print("Train:", X_train.shape)
print("Calibration:", X_cal.shape)
print("Test:", X_test.shape)


# ---------------------------------------------------------
# Verify Family is NOT part of the VAE input
# ---------------------------------------------------------
print()
print("Family included in VAE features:", FAMILY_COLUMN in FEATURE_COLUMNS)

assert FAMILY_COLUMN not in FEATURE_COLUMNS

print("Family ablation feature check: PASSED")


# ---------------------------------------------------------
# Compute deterministic reconstruction errors
# ---------------------------------------------------------
train_errors = reconstruction_errors(model, X_train)
cal_errors = reconstruction_errors(model, X_cal)
test_errors = reconstruction_errors(model, X_test)


# ---------------------------------------------------------
# Identify known / unseen test groups
# ---------------------------------------------------------
if FAMILY_COLUMN not in metadata.columns:
    raise ValueError("Family column missing from test metadata.")

locky_mask = metadata[FAMILY_COLUMN].astype(str).eq("Locky").to_numpy()
known_mask = ~locky_mask

known_errors = test_errors[known_mask]
locky_errors = test_errors[locky_mask]


# ---------------------------------------------------------
# Calibration threshold
# IMPORTANT:
# Only known calibration data is used.
# ---------------------------------------------------------
threshold = float(np.percentile(cal_errors, 95))

cal_rejected = cal_errors > threshold
known_test_rejected = known_errors > threshold
locky_rejected = locky_errors > threshold

known_rejection_rate = float(np.mean(known_test_rejected))
locky_unknown_rate = float(np.mean(locky_rejected))
locky_false_accept_rate = float(np.mean(~locky_rejected))
calibration_rejection_rate = float(np.mean(cal_rejected))


# ---------------------------------------------------------
# Reconstruction ROC-AUC
# Test only; no threshold fitting.
# ---------------------------------------------------------
y_open = np.concatenate(
    [
        np.zeros(len(known_errors), dtype=np.int64),
        np.ones(len(locky_errors), dtype=np.int64),
    ]
)

scores_open = np.concatenate(
    [
        known_errors,
        locky_errors,
    ]
)

roc_auc = float(roc_auc_score(y_open, scores_open))


# ---------------------------------------------------------
# Summaries
# ---------------------------------------------------------
summaries = [
    summarize("train", train_errors),
    summarize("calibration", cal_errors),
    summarize("known_test", known_errors),
    summarize("locky_test", locky_errors),
]


# ---------------------------------------------------------
# Comparison against Phase 8 reference
# ---------------------------------------------------------
phase8_reference = {
    "calibration_mean": 0.1246670559,
    "calibration_p95": 0.23906713426113124,
    "known_test_mean": 0.1252707243,
    "known_test_p95": 0.2411741,
    "locky_test_mean": 0.2711395025,
    "locky_test_p95": 0.4136152,
    "reconstruction_roc_auc": 0.8981559533,
    "threshold": 0.23906713426113124,
    "locky_unknown_rate": 0.583952,
    "locky_false_accept_rate": 0.416048,
    "known_rejection_rate": 0.052077,
}


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------
report = {
    "experiment": "E4b",
    "description": "VAE representation ablation without Family feature",
    "seed": SEED,
    "device": str(DEVICE),
    "checkpoint": str(CHECKPOINT.relative_to(PROJECT_ROOT)),
    "feature_columns": FEATURE_COLUMNS,
    "input_dim": len(FEATURE_COLUMNS),
    "latent_dim": LATENT_DIM,
    "train_size": int(len(X_train)),
    "calibration_size": int(len(X_cal)),
    "test_size": int(len(X_test)),
    "known_test_size": int(np.sum(known_mask)),
    "locky_test_size": int(np.sum(locky_mask)),
    "threshold_method": "95th percentile of known calibration reconstruction error",
    "threshold": threshold,
    "calibration_rejection_rate": calibration_rejection_rate,
    "known_rejection_rate": known_rejection_rate,
    "locky_unknown_detection_rate": locky_unknown_rate,
    "locky_false_accept_rate": locky_false_accept_rate,
    "reconstruction_roc_auc_known_vs_locky": roc_auc,
    "summaries": summaries,
    "phase8_family_included_reference": phase8_reference,
}


REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

import json

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)


# ---------------------------------------------------------
# Console output
# ---------------------------------------------------------
print()
print("=" * 70)
print("E4b VAE VALIDATION — FAMILY REMOVED")
print("=" * 70)

for item in summaries:
    print()
    print(item["name"])
    print("  count :", item["count"])
    print("  mean  :", item["mean"])
    print("  std   :", item["std"])
    print("  median:", item["median"])
    print("  p95   :", item["p95"])
    print("  p99   :", item["p99"])
    print("  max   :", item["max"])

print()
print("-" * 70)
print("OPEN-SET RECONSTRUCTION RESULTS")
print("-" * 70)
print("Calibration threshold:", threshold)
print("Calibration rejection:", calibration_rejection_rate)
print("Known test rejection :", known_rejection_rate)
print("Locky UNKNOWN rate    :", locky_unknown_rate)
print("Locky false-accept    :", locky_false_accept_rate)
print("ROC-AUC               :", roc_auc)

print()
print("-" * 70)
print("PHASE 8 REFERENCE — FAMILY INCLUDED")
print("-" * 70)
print("Threshold             :", phase8_reference["threshold"])
print("Known rejection       :", phase8_reference["known_rejection_rate"])
print("Locky UNKNOWN rate    :", phase8_reference["locky_unknown_rate"])
print("Locky false-accept    :", phase8_reference["locky_false_accept_rate"])
print("ROC-AUC               :", phase8_reference["reconstruction_roc_auc"])

print()
print("Report saved:", REPORT_PATH)
print("=" * 70)
