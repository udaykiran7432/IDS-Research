from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.vae.model import VAE


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase7_open_set"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "phase8"
    / "vae_family_heldout_seed_42.pt"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "phase8"

BATCH_SIZE = 256

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


def load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing Phase 7 split: {path}"
        )

    df = pd.read_csv(path)

    missing = [
        column
        for column in FEATURE_COLUMNS + [TARGET_COLUMN]
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{name} is missing columns: {missing}"
        )

    return df


def extract(
    model: VAE,
    X: np.ndarray,
    device: torch.device,
):
    dataset = TensorDataset(
        torch.from_numpy(X)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    latents = []
    errors = []

    model.eval()

    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)

            # Deterministic latent representation.
            mu, _ = model.encode(x)

            # Deterministic reconstruction.
            reconstruction = model.decode(mu)

            # Per-sample reconstruction MSE.
            error = torch.mean(
                (reconstruction - x) ** 2,
                dim=1,
            )

            latents.append(
                mu.cpu().numpy()
            )

            errors.append(
                error.cpu().numpy()
            )

    return (
        np.concatenate(latents, axis=0),
        np.concatenate(errors, axis=0),
    )


def main() -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
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

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    train_df = load_split("train")
    calibration_df = load_split("calibration")
    test_df = load_split("test")

    # Strict split checks.
    if (train_df["Family"] == -1).any():
        raise RuntimeError(
            "Train contains Family=-1."
        )

    if (calibration_df["Family"] == -1).any():
        raise RuntimeError(
            "Calibration contains Family=-1."
        )

    unseen_test_count = int(
        (test_df["Family"] == -1).sum()
    )

    if unseen_test_count != 25062:
        raise RuntimeError(
            "Unexpected unseen-family test count: "
            f"{unseen_test_count}"
        )

    X_train = train_df[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float32)

    X_calibration = calibration_df[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float32)

    X_test = test_df[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float32)

    for name, X in [
        ("train", X_train),
        ("calibration", X_calibration),
        ("test", X_test),
    ]:
        if not np.isfinite(X).all():
            raise RuntimeError(
                f"{name} contains non-finite features."
            )

    (
        train_latent,
        train_error,
    ) = extract(
        model,
        X_train,
        device,
    )

    (
        calibration_latent,
        calibration_error,
    ) = extract(
        model,
        X_calibration,
        device,
    )

    (
        test_latent,
        test_error,
    ) = extract(
        model,
        X_test,
        device,
    )

    # Separate the final test population using metadata encoded
    # in the frozen Phase 7 feature representation.
    test_unseen_mask = (
        test_df["Family"].to_numpy()
        == -1
    )

    test_known_mask = ~test_unseen_mask

    validation = {
        "experiment": (
            "Phase 8 family-held-out VAE validation"
        ),
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "device": str(device),
        "model": {
            "input_dim": checkpoint["input_dim"],
            "hidden_dim": checkpoint["hidden_dim"],
            "latent_dim": checkpoint["latent_dim"],
            "beta": checkpoint["beta"],
        },
        "splits": {
            "train": {
                "samples": int(len(train_df)),
                "family_minus_one": int(
                    (train_df["Family"] == -1).sum()
                ),
                "latent_shape": list(
                    train_latent.shape
                ),
                "reconstruction_error": summarize(
                    train_error
                ),
            },
            "calibration": {
                "samples": int(
                    len(calibration_df)
                ),
                "family_minus_one": int(
                    (
                        calibration_df["Family"]
                        == -1
                    ).sum()
                ),
                "latent_shape": list(
                    calibration_latent.shape
                ),
                "reconstruction_error": summarize(
                    calibration_error
                ),
            },
            "test": {
                "samples": int(len(test_df)),
                "known_samples": int(
                    test_known_mask.sum()
                ),
                "unseen_samples": int(
                    test_unseen_mask.sum()
                ),
                "latent_shape": list(
                    test_latent.shape
                ),
                "reconstruction_error": summarize(
                    test_error
                ),
            },
            "test_known_only": {
                "samples": int(
                    test_known_mask.sum()
                ),
                "reconstruction_error": summarize(
                    test_error[test_known_mask]
                ),
            },
            "test_unseen_locky": {
                "samples": int(
                    test_unseen_mask.sum()
                ),
                "reconstruction_error": summarize(
                    test_error[test_unseen_mask]
                ),
            },
        },
        "numerical_checks": {
            "train_latent_finite": bool(
                np.isfinite(train_latent).all()
            ),
            "calibration_latent_finite": bool(
                np.isfinite(
                    calibration_latent
                ).all()
            ),
            "test_latent_finite": bool(
                np.isfinite(test_latent).all()
            ),
            "train_error_finite": bool(
                np.isfinite(train_error).all()
            ),
            "calibration_error_finite": bool(
                np.isfinite(
                    calibration_error
                ).all()
            ),
            "test_error_finite": bool(
                np.isfinite(test_error).all()
            ),
            "train_error_nonnegative": bool(
                (train_error >= 0).all()
            ),
            "calibration_error_nonnegative": bool(
                (calibration_error >= 0).all()
            ),
            "test_error_nonnegative": bool(
                (test_error >= 0).all()
            ),
        },
    }

    report_path = (
        REPORT_DIR
        / "vae_validation_seed_42.json"
    )

    with report_path.open("w") as f:
        json.dump(
            validation,
            f,
            indent=2,
        )

    np.savez_compressed(
        REPORT_DIR
        / "vae_latent_representations_seed_42.npz",
        train_latent=train_latent,
        calibration_latent=calibration_latent,
        test_latent=test_latent,
        train_reconstruction_error=train_error,
        calibration_reconstruction_error=(
            calibration_error
        ),
        test_reconstruction_error=test_error,
    )

    print(
        "=== Phase 8 VAE Validation ==="
    )
    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )
    print(
        "Device    :",
        device,
    )
    print()

    for name, values in [
        ("Train", train_error),
        ("Calibration", calibration_error),
        (
            "Test known",
            test_error[test_known_mask],
        ),
        (
            "Test Locky",
            test_error[test_unseen_mask],
        ),
    ]:
        print(
            f"{name} reconstruction error:"
        )
        print(
            json.dumps(
                summarize(values),
                indent=2,
            )
        )
        print()

    print("Numerical checks:")
    print(
        json.dumps(
            validation[
                "numerical_checks"
            ],
            indent=2,
        )
    )

    print()
    print(
        "Validation report:",
        report_path,
    )


if __name__ == "__main__":
    main()
