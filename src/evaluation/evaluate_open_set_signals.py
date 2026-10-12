from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "evaluation"))

from src.vae.model import VAE
from controlled_agents import ControlledQAgent


SEED = 42
DEVICE = torch.device("cpu")
BATCH_SIZE = 4096
REJECTION_RATE = 0.05

FEATURES = [
    "Time", "Protcol", "Flag", "Clusters",
    "Threats", "USD", "BTC",
]

DATA_DIR = PROJECT_ROOT / "data/processed/phase7_open_set"
VAE_PATH = PROJECT_ROOT / "artifacts/phase10/vae_no_family_seed_42.pt"
AGENT_PATH = PROJECT_ROOT / "artifacts/phase10/e4b_d3qn_per_seed_42.pt"
OUTPUT_PATH = PROJECT_ROOT / "reports/phase10/open_set_signal_comparison_seed_42.json"


def make_representations(model, X):
    outputs = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            x = torch.as_tensor(
                X[start:start + BATCH_SIZE],
                dtype=torch.float32,
                device=DEVICE,
            )
            mu, _ = model.encode(x)
            reconstruction = model.decode(mu)
            error = ((reconstruction - x) ** 2).mean(dim=1, keepdim=True)
            outputs.append(
                torch.cat([mu, error], dim=1).cpu().numpy()
            )

    return np.concatenate(outputs).astype(np.float32)


def q_values(agent, X):
    outputs = []
    agent.policy_net.eval()

    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            batch = torch.as_tensor(
                X[start:start + BATCH_SIZE],
                dtype=torch.float32,
                device=DEVICE,
            )
            outputs.append(agent.policy_net(batch).cpu().numpy())

    return np.concatenate(outputs)


def percentile_scores(reference, values):
    """Empirical percentile of each value relative to known calibration."""
    reference = np.sort(np.asarray(reference))
    values = np.asarray(values)
    return np.searchsorted(reference, values, side="right") / len(reference)


def classification_metrics(y_true, y_pred):
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "accuracy": (tn + tp) / max(tn + fp + fn + tp, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def evaluate_variant(name, cal_score, test_score, predictions,
                    y_test, known_mask, locky_mask):
    threshold = float(np.quantile(cal_score, 1 - REJECTION_RATE))
    rejected = test_score > threshold

    known_rejected = rejected[known_mask]
    locky_rejected = rejected[locky_mask]
    retained_known = known_mask & ~rejected

    y_open = np.concatenate([
        np.zeros(int(known_mask.sum()), dtype=np.int64),
        np.ones(int(locky_mask.sum()), dtype=np.int64),
    ])
    open_scores = np.concatenate([
        test_score[known_mask],
        test_score[locky_mask],
    ])

    final_known_predictions = predictions[retained_known]
    final_known_labels = y_test[retained_known]

    return {
        "threshold": threshold,
        "calibration_rejection_rate": float(
            np.mean(cal_score > threshold)
        ),
        "known_test_rejection_rate": float(known_rejected.mean()),
        "known_test_acceptance_rate": float((~known_rejected).mean()),
        "locky_unknown_detection_rate": float(locky_rejected.mean()),
        "locky_false_accept_rate": float((~locky_rejected).mean()),
        "known_retained_classification": classification_metrics(
            final_known_labels, final_known_predictions
        ),
        "unknown_score_roc_auc": float(
            roc_auc_score(y_open, open_scores)
        ),
        "known_rejected_count": int(known_rejected.sum()),
        "locky_rejected_count": int(locky_rejected.sum()),
    }


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    cal_df = pd.read_csv(DATA_DIR / "calibration.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    metadata = pd.read_csv(DATA_DIR / "test_metadata.csv")

    if len(test_df) != len(metadata):
        raise RuntimeError("Test data and metadata lengths do not match.")

    if "Family" not in metadata.columns:
        raise RuntimeError("Family is missing from test metadata.")

    X_train = train_df[FEATURES].to_numpy(dtype=np.float32)
    X_cal = cal_df[FEATURES].to_numpy(dtype=np.float32)
    X_test = test_df[FEATURES].to_numpy(dtype=np.float32)

    for name, X in [
        ("train", X_train), ("calibration", X_cal), ("test", X_test)
    ]:
        if not np.isfinite(X).all():
            raise RuntimeError(f"Non-finite values in {name} features.")

    if (cal_df["Family"] == -1).any():
        raise RuntimeError("Calibration contains unseen-family encoding.")

    if not (metadata["Family"].astype(str) == "Locky").any():
        raise RuntimeError("Locky was not found in test metadata.")

    # Load the existing seven-feature VAE.
    vae_ckpt = torch.load(VAE_PATH, map_location=DEVICE, weights_only=False)
    if vae_ckpt["feature_columns"] != FEATURES or vae_ckpt["input_dim"] != 7:
        raise RuntimeError("VAE checkpoint feature configuration mismatch.")

    vae = VAE(
        input_dim=vae_ckpt["input_dim"],
        hidden_dim=vae_ckpt["hidden_dim"],
        latent_dim=vae_ckpt["latent_dim"],
    ).to(DEVICE)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae.eval()

    R_train = make_representations(vae, X_train)
    R_cal = make_representations(vae, X_cal)
    R_test = make_representations(vae, X_test)

    # Load the existing five-state E4b D3QN+PER checkpoint.
    agent_ckpt = torch.load(
        AGENT_PATH, map_location=DEVICE, weights_only=False
    )
    if agent_ckpt.get("experiment") != "E4b":
        raise RuntimeError("Unexpected agent checkpoint.")
    if agent_ckpt.get("state_dim") != 5:
        raise RuntimeError("Expected a five-dimensional agent state.")

    agent = ControlledQAgent(
        config="E3",
        state_dim=5,
        action_dim=2,
        seed=SEED,
        device=DEVICE,
    )
    agent.policy_net.load_state_dict(agent_ckpt["model_state_dict"])
    agent.policy_net.eval()

    q_cal = q_values(agent, R_cal)
    q_test = q_values(agent, R_test)
    pred_cal = np.argmax(q_cal, axis=1)
    pred_test = np.argmax(q_test, axis=1)

    # Match the existing E4b binary label mapping:
    # original Prediction 1 = benign; 0/2 = ransomware.
    y_cal = np.where(cal_df["Prediction"].to_numpy() == 1, 0, 1)
    y_test = np.where(test_df["Prediction"].to_numpy() == 1, 0, 1)

    # Signal A: VAE reconstruction error.
    reconstruction_cal = R_cal[:, 4]
    reconstruction_test = R_test[:, 4]

    # Signal B: nearest-neighbour distance in the four-dimensional
    # latent space, using known training representations only.
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(R_train[:, :4])
    latent_cal = nn.kneighbors(R_cal[:, :4], return_distance=True)[0][:, 0]
    latent_test = nn.kneighbors(R_test[:, :4], return_distance=True)[0][:, 0]

    # Signal C: smaller Q-value margin means more uncertain.
    q_margin_cal = -np.abs(q_cal[:, 1] - q_cal[:, 0])
    q_margin_test = -np.abs(q_test[:, 1] - q_test[:, 0])

    # Signal D: equally weighted calibration-percentile combination.
    # No weights or thresholds are selected using Locky test labels.
    cal_percentiles = np.column_stack([
        percentile_scores(reconstruction_cal, reconstruction_cal),
        percentile_scores(latent_cal, latent_cal),
        percentile_scores(q_margin_cal, q_margin_cal),
    ])
    test_percentiles = np.column_stack([
        percentile_scores(reconstruction_cal, reconstruction_test),
        percentile_scores(latent_cal, latent_test),
        percentile_scores(q_margin_cal, q_margin_test),
    ])
    combined_cal = cal_percentiles.mean(axis=1)
    combined_test = test_percentiles.mean(axis=1)

    locky_mask = metadata["Family"].astype(str).eq("Locky").to_numpy()
    known_mask = ~locky_mask

    # Calibration accuracy is reported as context; it does not determine
    # the open-set thresholds beyond the fixed 95th-percentile rule.
    report = {
        "experiment": "E4b open-set rejection signal comparison",
        "seed": SEED,
        "features": FEATURES,
        "family_included": False,
        "vae_checkpoint": str(VAE_PATH.relative_to(PROJECT_ROOT)),
        "agent_checkpoint": str(AGENT_PATH.relative_to(PROJECT_ROOT)),
        "train_size": int(len(train_df)),
        "calibration_size": int(len(cal_df)),
        "test_size": int(len(test_df)),
        "known_test_size": int(known_mask.sum()),
        "locky_test_size": int(locky_mask.sum()),
        "threshold_policy": "95th percentile of known calibration scores",
        "combined_score": "Equal-weight mean of calibration percentile ranks",
        "calibration_classification": classification_metrics(y_cal, pred_cal),
        "variants": {},
        "selection_note": (
            "Test results are for final evaluation only; no test-driven "
            "threshold, weight, or model selection was performed."
        ),
    }

    variants = {
        "reconstruction_error": (reconstruction_cal, reconstruction_test),
        "latent_nearest_neighbor_distance": (latent_cal, latent_test),
        "negative_q_margin": (q_margin_cal, q_margin_test),
        "combined_percentile_score": (combined_cal, combined_test),
    }

    for name, (cal_score, test_score) in variants.items():
        report["variants"][name] = evaluate_variant(
            name, cal_score, test_score, pred_test,
            y_test, known_mask, locky_mask,
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2))

    print("Open-set signal comparison complete.")
    print("Report:", OUTPUT_PATH)
    for name, metrics in report["variants"].items():
        print(
            f"{name}: "
            f"Locky detection={metrics['locky_unknown_detection_rate']:.4f}, "
            f"Locky false acceptance={metrics['locky_false_accept_rate']:.4f}, "
            f"known rejection={metrics['known_test_rejection_rate']:.4f}, "
            f"AUC={metrics['unknown_score_roc_auc']:.4f}"
        )


if __name__ == "__main__":
    main()
