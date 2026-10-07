from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
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

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "phase10"
REPORT_DIR = PROJECT_ROOT / "reports" / "phase10"

SEED = 42
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3

INPUT_DIM = 7
HIDDEN_DIM = 16
LATENT_DIM = 4
BETA = 1.0

FEATURE_COLUMNS = [
    "Time",
    "Protcol",
    "Flag",
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


def main() -> None:
    set_seed(SEED)

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=== Phase 10 E4b VAE Training ===")
    print("Training source:", TRAIN_PATH)
    print("Family feature : REMOVED")

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_PATH}"
        )

    df = pd.read_csv(TRAIN_PATH)

    required_columns = (
        FEATURE_COLUMNS
        + ["Family"]
        + [TARGET_COLUMN]
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    # Strict family-held-out training check.
    unseen_family_count = int(
        (df["Family"] == -1).sum()
    )

    if unseen_family_count != 0:
        raise RuntimeError(
            "E4b VAE training contains "
            f"{unseen_family_count} Family=-1 samples."
        )

    if len(df) != 69428:
        raise RuntimeError(
            "Unexpected training size: "
            f"{len(df)}; expected 69428."
        )

    X = df[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32
    )

    if X.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Expected {INPUT_DIM} input features; "
            f"found {X.shape[1]}."
        )

    if not np.isfinite(X).all():
        raise RuntimeError(
            "E4b training features contain "
            "non-finite values."
        )

    y = df[TARGET_COLUMN].to_numpy(
        dtype=np.int64
    )

    if not np.isin(y, [0, 1]).all():
        raise RuntimeError(
            "Training Prediction values must "
            "be binary 0/1."
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

            loss, reconstruction_loss, kl_loss = (
                vae_loss(
                    reconstruction,
                    x,
                    mu,
                    logvar,
                    beta=BETA,
                )
            )

            loss.backward()
            optimizer.step()

            batch_size = x.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            running_reconstruction += (
                reconstruction_loss.item()
                * batch_size
            )

            running_kl += (
                kl_loss.item() * batch_size
            )

            samples += batch_size

        epoch_loss = (
            running_loss / samples
        )

        epoch_reconstruction = (
            running_reconstruction / samples
        )

        epoch_kl = (
            running_kl / samples
        )

        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss,
                "reconstruction_loss": (
                    epoch_reconstruction
                ),
                "kl_loss": epoch_kl,
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"loss={epoch_loss:.6f} | "
            f"reconstruction="
            f"{epoch_reconstruction:.6f} | "
            f"KL={epoch_kl:.6f}"
        )

    checkpoint_path = (
        ARTIFACT_DIR
        / "vae_no_family_seed_42.pt"
    )

    torch.save(
        {
            "experiment": "E4b",
            "seed": SEED,
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "latent_dim": LATENT_DIM,
            "beta": BETA,
            "feature_columns": FEATURE_COLUMNS,
            "model_state_dict": model.state_dict(),
            "final_training": history[-1],
        },
        checkpoint_path,
    )

    report = {
        "experiment": "E4b VAE without Family feature",
        "seed": SEED,
        "train_rows": len(df),
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": LATENT_DIM,
        "beta": BETA,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "feature_columns": FEATURE_COLUMNS,
        "family_feature_removed": True,
        "family_minus_one_training_rows": (
            unseen_family_count
        ),
        "device": str(device),
        "history": history,
        "checkpoint": str(checkpoint_path),
    }

    report_path = (
        REPORT_DIR
        / "e4b_vae_training_seed_42.json"
    )

    with report_path.open("w") as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print("Training complete.")
    print("Checkpoint:", checkpoint_path)
    print("Report    :", report_path)


if __name__ == "__main__":
    main()