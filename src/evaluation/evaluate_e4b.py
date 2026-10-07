import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "evaluation"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "proposed"))

from src.vae.model import VAE
from controlled_agents import ControlledQAgent


SEED = 42

VAE_CHECKPOINT = PROJECT_ROOT / (
    "artifacts/phase10/vae_no_family_seed_42.pt"
)

AGENT_CHECKPOINT = PROJECT_ROOT / (
    "artifacts/phase10/e4b_d3qn_per_seed_42.pt"
)

CAL_PATH = PROJECT_ROOT / (
    "data/processed/phase7_open_set/calibration.csv"
)

TEST_PATH = PROJECT_ROOT / (
    "data/processed/phase7_open_set/test.csv"
)

METADATA_PATH = PROJECT_ROOT / (
    "data/processed/phase7_open_set/test_metadata.csv"
)

REPORT_PATH = PROJECT_ROOT / (
    "reports/phase10/e4b_evaluation_seed_42.json"
)

FEATURE_COLUMNS = [
    "Time",
    "Protcol",
    "Flag",
    "Clusters",
    "Threats",
    "USD",
    "BTC",
]

DEVICE = torch.device("cpu")


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def create_representation(model, X):
    model.eval()

    with torch.no_grad():
        x = torch.tensor(
            X,
            dtype=torch.float32,
            device=DEVICE,
        )

        mu, _ = model.encode(x)
        reconstruction = model.decode(mu)

        reconstruction_error = torch.mean(
            (reconstruction - x) ** 2,
            dim=1,
            keepdim=True,
        )

        representation = torch.cat(
            [mu, reconstruction_error],
            dim=1,
        )

    return representation.cpu().numpy().astype(np.float32)


def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    specificity = 1.0 - fpr

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "specificity": float(specificity),
        "confusion_matrix": [
            [tn, fp],
            [fn, tp],
        ],
    }


def predict(agent, X):
    predictions = []

    for state in X:
        action = agent.select_action(
            state,
            training=False,
        )
        predictions.append(action)

    return np.asarray(predictions, dtype=np.int64)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
cal_df = pd.read_csv(CAL_PATH)
test_df = pd.read_csv(TEST_PATH)
metadata = pd.read_csv(METADATA_PATH)

X_cal_raw = cal_df[FEATURE_COLUMNS].to_numpy(
    dtype=np.float32
)

X_test_raw = test_df[FEATURE_COLUMNS].to_numpy(
    dtype=np.float32
)

# Phase-7 mapping:
# Prediction 1 -> benign 0
# Prediction 0/2 -> ransomware 1
y_cal = np.where(
    cal_df["Prediction"].to_numpy() == 1,
    0,
    1,
).astype(np.int64)

y_test = np.where(
    test_df["Prediction"].to_numpy() == 1,
    0,
    1,
).astype(np.int64)

assert len(test_df) == len(metadata)


# ---------------------------------------------------------
# Load VAE
# ---------------------------------------------------------
vae_ckpt = torch.load(
    VAE_CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

assert vae_ckpt["input_dim"] == 7
assert "Family" not in vae_ckpt["feature_columns"]

vae = VAE(
    input_dim=vae_ckpt["input_dim"],
    hidden_dim=vae_ckpt["hidden_dim"],
    latent_dim=vae_ckpt["latent_dim"],
)

vae.load_state_dict(
    vae_ckpt["model_state_dict"]
)

vae.to(DEVICE)
vae.eval()


# ---------------------------------------------------------
# Represent calibration/test
# ---------------------------------------------------------
X_cal = create_representation(
    vae,
    X_cal_raw,
)

X_test = create_representation(
    vae,
    X_test_raw,
)


# ---------------------------------------------------------
# Reconstruction threshold
#
# IMPORTANT:
# calibration only.
# ---------------------------------------------------------
cal_reconstruction_error = X_cal[:, 4]

threshold = float(
    np.percentile(
        cal_reconstruction_error,
        95,
    )
)

print("Calibration threshold:", threshold)


# ---------------------------------------------------------
# Load E4b D3QN + PER
# ---------------------------------------------------------
agent_ckpt = torch.load(
    AGENT_CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

assert agent_ckpt["experiment"] == "E4b"
assert agent_ckpt["state_dim"] == 5

agent = ControlledQAgent(
    config="E3",
    state_dim=5,
    action_dim=2,
    seed=SEED,
    device=DEVICE,
)

agent.policy_net.load_state_dict(
    agent_ckpt["model_state_dict"]
)

agent.target_net.load_state_dict(
    agent_ckpt["target_state_dict"]
)

agent.policy_net.eval()
agent.target_net.eval()

agent.epsilon = 0.0


# ---------------------------------------------------------
# D3QN predictions
# ---------------------------------------------------------
cal_predictions = predict(
    agent,
    X_cal,
)

test_predictions = predict(
    agent,
    X_test,
)


# ---------------------------------------------------------
# Calibration classification metrics
# ---------------------------------------------------------
cal_metrics = classification_metrics(
    y_cal,
    cal_predictions,
)


# ---------------------------------------------------------
# Open-set decision
# ---------------------------------------------------------
test_reconstruction_error = X_test[:, 4]

unknown_mask = test_reconstruction_error > threshold

known_prediction_mask = ~unknown_mask

# Final system:
# reconstruction threshold decides UNKNOWN.
# Otherwise use D3QN classification.
final_predictions = test_predictions.copy()

# Use -1 to represent UNKNOWN.
final_predictions[unknown_mask] = -1


# ---------------------------------------------------------
# Test groups
# ---------------------------------------------------------
locky_mask = (
    metadata["Family"]
    .astype(str)
    .eq("Locky")
    .to_numpy()
)

known_mask = ~locky_mask

assert np.sum(locky_mask) == 25062
assert np.sum(known_mask) == 37195


# ---------------------------------------------------------
# Open-set statistics
# ---------------------------------------------------------
known_unknown = int(
    np.sum(unknown_mask & known_mask)
)

known_accepted = int(
    np.sum(known_prediction_mask & known_mask)
)

locky_unknown = int(
    np.sum(unknown_mask & locky_mask)
)

locky_accepted = int(
    np.sum(known_prediction_mask & locky_mask)
)

known_rejection_rate = (
    known_unknown / np.sum(known_mask)
)

known_acceptance_rate = (
    known_accepted / np.sum(known_mask)
)

locky_unknown_rate = (
    locky_unknown / np.sum(locky_mask)
)

locky_false_accept_rate = (
    locky_accepted / np.sum(locky_mask)
)


# ---------------------------------------------------------
# Known closed-set classification
# Ignore UNKNOWN for this metric.
# ---------------------------------------------------------
known_y_true = y_test[known_mask]
known_dqn_pred = test_predictions[known_mask]

known_closed_set_metrics = classification_metrics(
    known_y_true,
    known_dqn_pred,
)


# ---------------------------------------------------------
# Known retained classification
# Only known traffic accepted by open-set gate.
# ---------------------------------------------------------
retained_mask = known_mask & known_prediction_mask

retained_y_true = y_test[retained_mask]
retained_predictions = test_predictions[retained_mask]

known_retained_metrics = classification_metrics(
    retained_y_true,
    retained_predictions,
) if len(retained_y_true) > 0 else None


# ---------------------------------------------------------
# Locky accepted classification
# This shows how unseen Locky is being forced
# into known classes when not rejected.
# ---------------------------------------------------------
locky_accepted_mask = locky_mask & known_prediction_mask

locky_accepted_predictions = (
    test_predictions[locky_accepted_mask]
)

locky_accepted_counts = {
    "predicted_benign": int(
        np.sum(locky_accepted_predictions == 0)
    ),
    "predicted_ransomware": int(
        np.sum(locky_accepted_predictions == 1)
    ),
}


# ---------------------------------------------------------
# Calibration rejection
# ---------------------------------------------------------
calibration_rejection_rate = float(
    np.mean(
        cal_reconstruction_error > threshold
    )
)


# ---------------------------------------------------------
# Reconstruction summaries
# ---------------------------------------------------------
def summary(errors):
    return {
        "count": int(len(errors)),
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95)),
        "max": float(np.max(errors)),
    }


reconstruction_summary = {
    "calibration": summary(
        cal_reconstruction_error
    ),
    "known_test": summary(
        test_reconstruction_error[known_mask]
    ),
    "locky_test": summary(
        test_reconstruction_error[locky_mask]
    ),
}


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------
report = {
    "experiment": "E4b",
    "seed": SEED,
    "vae_features": FEATURE_COLUMNS,
    "family_included": False,
    "threshold_method": (
        "95th percentile of known calibration "
        "reconstruction error"
    ),
    "threshold": threshold,
    "calibration_rejection_rate": (
        calibration_rejection_rate
    ),
    "calibration_classification": cal_metrics,
    "test_size": int(len(test_df)),
    "known_test_size": int(np.sum(known_mask)),
    "locky_test_size": int(np.sum(locky_mask)),
    "known_unknown": known_unknown,
    "known_accepted": known_accepted,
    "known_rejection_rate": float(
        known_rejection_rate
    ),
    "known_acceptance_rate": float(
        known_acceptance_rate
    ),
    "locky_unknown": locky_unknown,
    "locky_accepted": locky_accepted,
    "locky_unknown_detection_rate": float(
        locky_unknown_rate
    ),
    "locky_false_accept_rate": float(
        locky_false_accept_rate
    ),
    "known_closed_set": known_closed_set_metrics,
    "known_retained": known_retained_metrics,
    "locky_accepted_classification": (
        locky_accepted_counts
    ),
    "reconstruction_summary": reconstruction_summary,
    "checkpoint": str(
        AGENT_CHECKPOINT.relative_to(PROJECT_ROOT)
    ),
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
print()
print("=" * 70)
print("E4b D3QN + PER + VAE EVALUATION")
print("=" * 70)

print()
print("Calibration:")
print("  threshold:", threshold)
print(
    "  rejection:",
    calibration_rejection_rate,
)

print()
print("Known test — closed set:")
for key, value in known_closed_set_metrics.items():
    print(f"  {key}: {value}")

print()
print("Known test — open-set gate:")
print("  UNKNOWN:", known_unknown)
print("  accepted:", known_accepted)
print("  rejection rate:", known_rejection_rate)
print("  acceptance rate:", known_acceptance_rate)

if known_retained_metrics is not None:
    print()
    print("Known retained — after UNKNOWN rejection:")
    for key, value in known_retained_metrics.items():
        print(f"  {key}: {value}")

print()
print("Locky unseen test:")
print("  UNKNOWN:", locky_unknown)
print("  accepted:", locky_accepted)
print("  UNKNOWN detection rate:", locky_unknown_rate)
print("  false-accept rate:", locky_false_accept_rate)

print()
print("Locky accepted-class predictions:")
print(" ", locky_accepted_counts)

print()
print("Reconstruction:")
print(" ", reconstruction_summary)

print()
print("Report saved:", REPORT_PATH)
print("=" * 70)
