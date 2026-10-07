from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.proposed.data import load_data
from src.vae.model import VAE


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT / "artifacts" / "phase6" / "vae_seed_42.pt"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "phase6"

BATCH_SIZE = 256


def summarize(values: np.ndarray) -> dict:
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def extract(
    model: VAE,
    X: np.ndarray,
    device: torch.device,
):
    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    latents = []
    errors = []
    reconstructions = []

    model.eval()

    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)

            # Deterministic latent representation.
            mu, _ = model.encode(x)

            # Deterministic reconstruction from latent mean.
            reconstruction = model.decode(mu)

            # Per-sample mean squared reconstruction error.
            error = torch.mean(
                (reconstruction - x) ** 2,
                dim=1,
            )

            latents.append(mu.cpu().numpy())
            errors.append(error.cpu().numpy())
            reconstructions.append(reconstruction.cpu().numpy())

    return (
        np.concatenate(latents, axis=0),
        np.concatenate(errors, axis=0),
        np.concatenate(reconstructions, axis=0),
    )


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model = VAE(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        latent_dim=checkpoint["latent_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    X_train, _, X_test, _, _, _ = load_data()

    (
        train_latent,
        train_errors,
        train_reconstruction,
    ) = extract(
        model,
        X_train,
        device,
    )

    (
        test_latent,
        test_errors,
        test_reconstruction,
    ) = extract(
        model,
        X_test,
        device,
    )

    train_feature_mse = np.mean(
        (train_reconstruction - X_train) ** 2,
        axis=0,
    )

    test_feature_mse = np.mean(
        (test_reconstruction - X_test) ** 2,
        axis=0,
    )

    validation = {
        "experiment": "Phase 6 VAE validation",
        "checkpoint": str(CHECKPOINT_PATH),
        "device": str(device),
        "model": {
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "latent_dim": model.latent_dim,
            "beta": checkpoint["beta"],
        },
        "train": {
            "samples": int(len(X_train)),
            "latent_shape": list(train_latent.shape),
            "reconstruction_error": summarize(train_errors),
            "latent_mean": train_latent.mean(axis=0).tolist(),
            "latent_std": train_latent.std(axis=0).tolist(),
            "latent_min": train_latent.min(axis=0).tolist(),
            "latent_max": train_latent.max(axis=0).tolist(),
            "feature_mse": train_feature_mse.tolist(),
        },
        "test": {
            "samples": int(len(X_test)),
            "latent_shape": list(test_latent.shape),
            "reconstruction_error": summarize(test_errors),
            "latent_mean": test_latent.mean(axis=0).tolist(),
            "latent_std": test_latent.std(axis=0).tolist(),
            "latent_min": test_latent.min(axis=0).tolist(),
            "latent_max": test_latent.max(axis=0).tolist(),
            "feature_mse": test_feature_mse.tolist(),
        },
        "numerical_checks": {
            "train_latent_finite": bool(
                np.isfinite(train_latent).all()
            ),
            "test_latent_finite": bool(
                np.isfinite(test_latent).all()
            ),
            "train_error_finite": bool(
                np.isfinite(train_errors).all()
            ),
            "test_error_finite": bool(
                np.isfinite(test_errors).all()
            ),
            "train_error_nonnegative": bool(
                (train_errors >= 0).all()
            ),
            "test_error_nonnegative": bool(
                (test_errors >= 0).all()
            ),
        },
    }

    output_path = REPORT_DIR / "vae_validation.json"

    with output_path.open("w") as f:
        json.dump(validation, f, indent=2)

    np.savez_compressed(
        REPORT_DIR / "vae_latent_representations.npz",
        train_latent=train_latent,
        test_latent=test_latent,
        train_reconstruction_error=train_errors,
        test_reconstruction_error=test_errors,
    )

    print("=== Phase 6 VAE Validation ===")
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Device    :", device)
    print()
    print("Train latent shape:", train_latent.shape)
    print("Test latent shape :", test_latent.shape)
    print()
    print("Train reconstruction error:")
    print(json.dumps(
        validation["train"]["reconstruction_error"],
        indent=2,
    ))
    print()
    print("Test reconstruction error:")
    print(json.dumps(
        validation["test"]["reconstruction_error"],
        indent=2,
    ))
    print()
    print("Numerical checks:")
    print(json.dumps(
        validation["numerical_checks"],
        indent=2,
    ))
    print()
    print("Validation report:", output_path)


if __name__ == "__main__":
    main()
