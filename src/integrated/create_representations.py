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
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase8_integrated"
)

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

REPRESENTATION_COLUMNS = [
    "latent_0",
    "latent_1",
    "latent_2",
    "latent_3",
    "reconstruction_error",
]


def load_model(device: torch.device) -> VAE:
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

    model.eval()

    return model


def transform(
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

    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)

            mu, _ = model.encode(x)
            reconstruction = model.decode(mu)

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


def process_split(
    name: str,
    model: VAE,
    device: torch.device,
):
    input_path = DATA_DIR / f"{name}.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing input split: {input_path}"
        )

    df = pd.read_csv(input_path)

    X = df[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float32)

    if not np.isfinite(X).all():
        raise RuntimeError(
            f"{name} contains non-finite input features."
        )

    latent, reconstruction_error = transform(
        model,
        X,
        device,
    )

    output = pd.DataFrame(
        latent,
        columns=[
            "latent_0",
            "latent_1",
            "latent_2",
            "latent_3",
        ],
    )

    output["reconstruction_error"] = (
        reconstruction_error
    )

    output[TARGET_COLUMN] = df[
        TARGET_COLUMN
    ].to_numpy(dtype=np.int64)

    output_path = (
        OUTPUT_DIR / f"{name}.csv"
    )

    output.to_csv(
        output_path,
        index=False,
    )

    return {
        "input_samples": int(len(df)),
        "output_samples": int(len(output)),
        "latent_shape": list(latent.shape),
        "reconstruction_error": {
            "mean": float(
                reconstruction_error.mean()
            ),
            "std": float(
                reconstruction_error.std()
            ),
            "min": float(
                reconstruction_error.min()
            ),
            "p95": float(
                np.percentile(
                    reconstruction_error,
                    95,
                )
            ),
            "max": float(
                reconstruction_error.max()
            ),
        },
        "output": str(output_path),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = load_model(device)

    results = {
        "experiment": (
            "Phase 8 VAE representation extraction"
        ),
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "device": str(device),
        "representation_columns": (
            REPRESENTATION_COLUMNS
        ),
        "source_feature_columns": (
            FEATURE_COLUMNS
        ),
        "splits": {},
    }

    for name in [
        "train",
        "calibration",
        "test",
    ]:
        print(
            f"Processing {name}..."
        )

        results["splits"][name] = (
            process_split(
                name,
                model,
                device,
            )
        )

        print(
            "  samples:",
            results["splits"][name][
                "output_samples"
            ],
        )

    # Explicit dimensionality check.
    for name in [
        "train",
        "calibration",
        "test",
    ]:
        path = OUTPUT_DIR / f"{name}.csv"
        df = pd.read_csv(path)

        if list(df.columns) != (
            REPRESENTATION_COLUMNS
            + [TARGET_COLUMN]
        ):
            raise RuntimeError(
                f"Unexpected columns in {path}: "
                f"{list(df.columns)}"
            )

        values = df[
            REPRESENTATION_COLUMNS
        ].to_numpy(dtype=np.float64)

        if not np.isfinite(values).all():
            raise RuntimeError(
                f"Non-finite representation values "
                f"in {path}"
            )

    report_path = (
        REPORT_DIR
        / "representation_extraction_seed_42.json"
    )

    with report_path.open("w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print(
        "=== Phase 8 Representation Extraction ==="
    )
    print(
        "Representation dimension: 5"
    )
    print(
        "Columns:",
        REPRESENTATION_COLUMNS,
    )

    for name, result in (
        results["splits"].items()
    ):
        print(
            f"{name}: "
            f"{result['output_samples']} samples"
        )

    print()
    print(
        "Validation: all representation values finite."
    )
    print(
        "Report:",
        report_path,
    )


if __name__ == "__main__":
    main()
