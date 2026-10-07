from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.proposed.data import load_data
from src.vae.model import VAE, vae_loss


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "phase6"
REPORT_DIR = PROJECT_ROOT / "reports" / "phase6"

SEED = 42
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3

INPUT_DIM = 8
HIDDEN_DIM = 16
LATENT_DIM = 4
BETA = 1.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(False)


def get_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def evaluate(
    model: VAE,
    loader: DataLoader,
    device: torch.device,
):
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
            total_reconstruction += reconstruction_loss.item() * batch_size
            total_kl += kl_loss.item() * batch_size
            total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "reconstruction_loss": (
            total_reconstruction / total_samples
        ),
        "kl_loss": total_kl / total_samples,
    }


def main():
    set_seed(SEED)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()

    X_train, _, X_test, _, _, _ = load_data()

    train_tensor = torch.from_numpy(X_train)
    test_tensor = torch.from_numpy(X_test)

    train_dataset = TensorDataset(train_tensor)
    test_dataset = TensorDataset(test_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
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

    history = []

    print("=== Phase 6 VAE Training ===")
    print("Device       :", device)
    print("Seed         :", SEED)
    print("Train samples:", len(train_dataset))
    print("Test samples :", len(test_dataset))
    print("Input dim    :", INPUT_DIM)
    print("Hidden dim   :", HIDDEN_DIM)
    print("Latent dim   :", LATENT_DIM)
    print("Beta         :", BETA)
    print("Batch size   :", BATCH_SIZE)
    print("Epochs       :", EPOCHS)
    print()

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
            running_kl += kl_loss.item() * batch_size
            samples += batch_size

        train_epoch = {
            "loss": running_loss / samples,
            "reconstruction_loss": (
                running_reconstruction / samples
            ),
            "kl_loss": running_kl / samples,
        }

        train_eval = evaluate(
            model,
            train_eval_loader,
            device,
        )

        test_eval = evaluate(
            model,
            test_loader,
            device,
        )

        epoch_record = {
            "epoch": epoch,
            "train": train_epoch,
            "train_eval": train_eval,
            "test_eval": test_eval,
        }

        history.append(epoch_record)

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train loss={train_epoch['loss']:.6f} | "
            f"train recon={train_epoch['reconstruction_loss']:.6f} | "
            f"train KL={train_epoch['kl_loss']:.6f} | "
            f"test recon={test_eval['reconstruction_loss']:.6f}"
        )

    checkpoint_path = ARTIFACT_DIR / "vae_seed_42.pt"

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
        },
        checkpoint_path,
    )

    history_path = REPORT_DIR / "vae_training_history.json"

    with history_path.open("w") as f:
        json.dump(history, f, indent=2)

    summary = {
        "experiment": "Phase 6 VAE",
        "seed": SEED,
        "device": str(device),
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": LATENT_DIM,
        "beta": BETA,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "checkpoint": str(checkpoint_path),
        "final_train": history[-1]["train_eval"],
        "final_test": history[-1]["test_eval"],
    }

    summary_path = REPORT_DIR / "vae_training_summary.json"

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("Training complete.")
    print("Checkpoint:", checkpoint_path)
    print("History   :", history_path)
    print("Summary   :", summary_path)


if __name__ == "__main__":
    main()
