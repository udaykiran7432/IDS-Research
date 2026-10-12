from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.vae.model import VAE

DEFAULT_SEED = 42
SEED = DEFAULT_SEED
DEVICE = torch.device("cpu")
BATCH_SIZE = 4096
TARGET_REJECTION_RATE = 0.05

FEATURES = [
    "Time", "Protcol", "Flag", "Clusters",
    "Threats", "USD", "BTC",
]

DATA_DIR = PROJECT_ROOT / "data/processed/phase7_open_set"
VAE_PATH = PROJECT_ROOT / f"artifacts/phase10/vae_no_family_seed_{SEED}.pt"
OUTPUT_PATH = PROJECT_ROOT / f"reports/phase10/latent_distance_ablation_seed_{SEED}.json"


def representations(model, X):
    latent_parts = []
    error_parts = []
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
            error = ((reconstruction - x) ** 2).mean(dim=1)
            latent_parts.append(mu.cpu().numpy())
            error_parts.append(error.cpu().numpy())

    return (
        np.concatenate(latent_parts).astype(np.float64),
        np.concatenate(error_parts).astype(np.float64),
    )


def open_set_metrics(cal_scores, test_scores, unknown_mask):
    threshold = float(
        np.quantile(cal_scores, 1.0 - TARGET_REJECTION_RATE)
    )
    rejected = test_scores > threshold
    known_mask = ~unknown_mask

    known_rejection = float(rejected[known_mask].mean())
    unknown_detection = float(rejected[unknown_mask].mean())

    labels = unknown_mask.astype(np.int64)
    auc = float(roc_auc_score(labels, test_scores))

    return {
        "threshold": threshold,
        "calibration_rejection_rate": float(
            (cal_scores > threshold).mean()
        ),
        "known_test_rejection_rate": known_rejection,
        "known_test_acceptance_rate": 1.0 - known_rejection,
        "locky_unknown_detection_rate": unknown_detection,
        "locky_false_accept_rate": 1.0 - unknown_detection,
        "unknown_score_roc_auc": auc,
        "known_rejected_count": int(rejected[known_mask].sum()),
        "locky_rejected_count": int(rejected[unknown_mask].sum()),
    }


def main():
    global SEED, VAE_PATH, OUTPUT_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="VAE seed to evaluate (default: 42)",
    )
    args = parser.parse_args()
    SEED = args.seed
    VAE_PATH = PROJECT_ROOT / f"artifacts/phase10/vae_no_family_seed_{SEED}.pt"
    OUTPUT_PATH = PROJECT_ROOT / f"reports/phase10/latent_distance_ablation_seed_{SEED}.json"

    if not VAE_PATH.is_file():
        raise FileNotFoundError(f"VAE checkpoint not found: {VAE_PATH}")

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

    locky_mask = metadata["Family"].astype(str).eq("Locky").to_numpy()
    if not locky_mask.any() or locky_mask.all():
        raise RuntimeError("Expected both Locky and known-family test records.")

    # This experiment is specifically Locky-held-out. Confirm no Locky
    # records occur in the known training or calibration partitions.
    if (train_df["Family"] == -1).any():
        raise RuntimeError("Unexpected unseen-family encoding in training.")
    if (cal_df["Family"] == -1).any():
        raise RuntimeError("Unexpected unseen-family encoding in calibration.")

    X_train = train_df[FEATURES].to_numpy(dtype=np.float32)
    X_cal = cal_df[FEATURES].to_numpy(dtype=np.float32)
    X_test = test_df[FEATURES].to_numpy(dtype=np.float32)

    for name, X in [
        ("train", X_train),
        ("calibration", X_cal),
        ("test", X_test),
    ]:
        if not np.isfinite(X).all():
            raise RuntimeError(f"Non-finite feature values in {name}.")

    checkpoint = torch.load(
        VAE_PATH, map_location=DEVICE, weights_only=False
    )
    if checkpoint["feature_columns"] != FEATURES:
        raise RuntimeError("VAE feature configuration mismatch.")
    if checkpoint["input_dim"] != len(FEATURES):
        raise RuntimeError("VAE input dimension mismatch.")

    model = VAE(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        latent_dim=checkpoint["latent_dim"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    Z_train, E_train = representations(model, X_train)
    Z_cal, E_cal = representations(model, X_cal)
    Z_test, E_test = representations(model, X_test)

    # Method 1: existing VAE reconstruction-error baseline.
    scores = {
        "reconstruction_error": (E_cal, E_test),
    }

    # Method 2: standardized Euclidean nearest-neighbour distance.
    # Scaling parameters are fitted using training latent vectors only.
    latent_scale = Z_train.std(axis=0, ddof=1)
    latent_scale = np.maximum(latent_scale, 1e-8)

    Z_train_std = Z_train / latent_scale
    Z_cal_std = Z_cal / latent_scale
    Z_test_std = Z_test / latent_scale

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
    nn.fit(Z_train_std)

    nn_cal = nn.kneighbors(Z_cal_std, return_distance=True)[0][:, 0]
    nn_test = nn.kneighbors(Z_test_std, return_distance=True)[0][:, 0]
    scores["standardized_euclidean_nn"] = (nn_cal, nn_test)

    # Method 3: regularized Mahalanobis distance from the training
    # latent distribution. Ledoit-Wolf covariance is fitted on train only.
    covariance = LedoitWolf().fit(Z_train)
    mahal_cal = np.maximum(covariance.mahalanobis(Z_cal), 0.0)
    mahal_test = np.maximum(covariance.mahalanobis(Z_test), 0.0)
    scores["regularized_mahalanobis"] = (mahal_cal, mahal_test)

    report = {
        "experiment": "Latent-space distance ablation",
        "seed": SEED,
        "features": FEATURES,
        "family_included_as_input": False,
        "unseen_test_family": "Locky",
        "train_size": int(len(train_df)),
        "calibration_size": int(len(cal_df)),
        "test_size": int(len(test_df)),
        "known_test_size": int((~locky_mask).sum()),
        "locky_test_size": int(locky_mask.sum()),
        "target_calibration_rejection_rate": TARGET_REJECTION_RATE,
        "threshold_policy": (
            "Per-method quantile of known calibration scores; "
            "no Locky test labels used for fitting or threshold selection"
        ),
        "methods": {},
        "selection_note": (
            "Single-seed evaluation. Test metrics are for reporting, "
            "not model or threshold selection. Multi-seed validation "
            "is required before drawing conclusions."
        ),
    }

    for name, (cal_scores, test_scores) in scores.items():
        report["methods"][name] = open_set_metrics(
            cal_scores, test_scores, locky_mask
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2))

    print("Latent distance ablation complete.")
    print("Report:", OUTPUT_PATH)
    for name, result in report["methods"].items():
        print(
            f"{name}: "
            f"Locky detection={result['locky_unknown_detection_rate']:.4f}, "
            f"Locky false acceptance={result['locky_false_accept_rate']:.4f}, "
            f"known rejection={result['known_test_rejection_rate']:.4f}, "
            f"AUC={result['unknown_score_roc_auc']:.4f}"
        )


if __name__ == "__main__":
    main()
