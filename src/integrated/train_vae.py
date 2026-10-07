from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.vae.model import VAE, vae_loss


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase7_open_set"
    / "train.csv"
)

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "phase8"
REPORT_DIR = PROJECT_ROOT / "reports" / "phase8"

SEED = 42
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3

INPUT_DIM = 8
HIDDEN_DIM = 16
LATENT_DIM = 4
BETA = 1.0

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(False)


def get_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )


def evaluate_training_reconstruction(
    model: VAE,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()

    total_loss = 0.0
    total_reconstruction = 0.0
    total_kl = 0.0
    total_samples = 0

    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)

            reconstruction, mu, logvar, _ = model(x)

            loss, reconstruction_loss, kl_loss = vae_loss(
                reconstruction,
                x,
                mu,
                logvar,
                beta=BETA,
            )

            batch_size = x.size(0)

            total_loss += loss.item() * batch_size
            total_reconstruction += (
                reconstruction_loss.item() * batch_size
            )
            total_kl += kl_loss.item() * batch_size
            total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "reconstruction_loss": (
            total_reconstruction / total_samples
        ),
        "kl_loss": total_kl / total_samples,
    }


def main() -> None:
    set_seed(SEED)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Phase 8 training data not found: {TRAIN_PATH}"
        )

    print("=== Phase 8 VAE Training ===")
    print("Training source:", TRAIN_PATH)

    df = pd.read_csv(TRAIN_PATH)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    # Critical family-held-out leakage check.
    unseen_family_count = int(
        (df["Family"] == -1).sum()
    )

    if unseen_family_count != 0:
        raise RuntimeError(
            "Phase 8 VAE training contains "
            f"{unseen_family_count} Family=-1 samples."
        )

    if len(df) != 69428:
        raise RuntimeError(
            "Unexpected Phase 8 training size: "
            f"{len(df)}; expected 69428."
        )

    X = df[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(X).all():
        raise RuntimeError(
            "Phase 8 VAE training features contain "
            "non-finite values."
        )

    if X.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Expected {INPUT_DIM} input features; "
            f"found {X.shape[1]}."
        )

    # The Prediction column is intentionally not used by the VAE.
    y = df[TARGET_COLUMN].to_numpy(dtype=np.int64)

    if not np.isin(y, [0, 1]).all():
        raise RuntimeError(
            "Training Prediction values must be binary 0/1."
        )

    device = get_device()

    dataset = TensorDataset(
        torch.from_numpy(X)
    )

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    train_eval_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = VAE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("Device       :", device)
    print("Seed         :", SEED)
    print("Train samples:", len(dataset))
    print("Family=-1    :", unseen_family_count)
    print("Input dim    :", INPUT_DIM)
    print("Hidden dim   :", HIDDEN_DIM)
    print("Latent dim   :", LATENT_DIM)
    print("Beta         :", BETA)
    print("Batch size   :", BATCH_SIZE)
    print("Epochs       :", EPOCHS)
    print()

    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        running_reconstruction = 0.0
        running_kl = 0.0
        samples = 0

        for (x,) in train_loader:
            x = x.to(device)

            optimizer.zero_grad()

            reconstruction, mu, logvar, _ = model(x)

            loss, reconstruction_loss, kl_loss = vae_loss(
                reconstruction,
                x,
                mu,
                logvar,
                beta=BETA,
            )

            loss.backward()
            optimizer.step()

            batch_size = x.size(0)

            running_loss += loss.item() * batch_size
            running_reconstruction += (
                reconstruction_loss.item() * batch_size
            )
            running_kl += (
                kl_loss.item() * batch_size
            )
            samples += batch_size

        train_epoch = {
            "loss": running_loss / samples,
            "reconstruction_loss": (
                running_reconstruction / samples
            ),
            "kl_loss": running_kl / samples,
        }

        train_eval = evaluate_training_reconstruction(
            model,
            train_eval_loader,
            device,
        )

        record = {
            "epoch": epoch,
            "train": train_epoch,
            "train_eval": train_eval,
        }

        history.append(record)

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train loss={train_epoch['loss']:.6f} | "
            f"train recon="
            f"{train_epoch['reconstruction_loss']:.6f} | "
            f"train KL="
            f"{train_epoch['kl_loss']:.6f}"
        )

    checkpoint_path = (
        ARTIFACT_DIR / "vae_family_heldout_seed_42.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "latent_dim": LATENT_DIM,
            "beta": BETA,
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "feature_columns": FEATURE_COLUMNS,
            "training_dataset": str(TRAIN_PATH),
            "training_samples": len(dataset),
            "unseen_family_training_samples": unseen_family_count,
        },
        checkpoint_path,
    )

    history_path = (
        REPORT_DIR / "vae_training_history_seed_42.json"
    )

    with history_path.open("w") as f:
        json.dump(history, f, indent=2)

    summary = {
        "experiment": (
            "Phase 8 family-held-out VAE"
        ),
        "seed": SEED,
        "device": str(device),
        "training_dataset": str(TRAIN_PATH),
        "train_samples": len(dataset),
        "unseen_family_training_samples": (
            unseen_family_count
        ),
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": LATENT_DIM,
        "beta": BETA,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "checkpoint": str(checkpoint_path),
        "final_train": history[-1]["train_eval"],
        "data_leakage_checks": {
            "family_minus_one_absent": (
                unseen_family_count == 0
            ),
            "training_size_correct": (
                len(dataset) == 69428
            ),
            "calibration_used_for_training": False,
            "test_used_for_training": False,
        },
    }

    summary_path = (
        REPORT_DIR / "vae_training_summary_seed_42.json"
    )

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("Training complete.")
    print("Checkpoint:", checkpoint_path)
    print("History   :", history_path)
    print("Summary   :", summary_path)


if __name__ == "__main__":
    main()
