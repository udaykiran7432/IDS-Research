from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = PROJECT_ROOT / "reports" / "phase8"

VALIDATION_PATH = (
    REPORT_DIR / "vae_validation_seed_42.json"
)

OUTPUT_PATH = (
    REPORT_DIR / "vae_open_set_calibration_seed_42.json"
)

KNOWN_REJECTION_RATE = 0.05


def main() -> None:
    with VALIDATION_PATH.open() as f:
        validation = json.load(f)

    calibration = validation["splits"]["calibration"][
        "reconstruction_error"
    ]

    # We need the individual calibration errors, not just
    # summary statistics. Load them from the saved NPZ.
    npz_path = (
        REPORT_DIR
        / "vae_latent_representations_seed_42.npz"
    )

    data = np.load(npz_path)

    calibration_errors = np.asarray(
        data["calibration_reconstruction_error"],
        dtype=np.float64,
    )

    if not np.isfinite(calibration_errors).all():
        raise RuntimeError(
            "Calibration reconstruction errors contain "
            "non-finite values."
        )

    if (calibration_errors < 0).any():
        raise RuntimeError(
            "Calibration reconstruction errors contain "
            "negative values."
        )

    threshold = float(
        np.quantile(
            calibration_errors,
            1.0 - KNOWN_REJECTION_RATE,
        )
    )

    rejected = calibration_errors >= threshold

    actual_rejection_rate = float(
        rejected.mean()
    )

    result = {
        "experiment": (
            "Phase 8 VAE reconstruction-error "
            "open-set calibration"
        ),
        "calibration_samples": int(
            len(calibration_errors)
        ),
        "selection_method": (
            "95th percentile of known calibration "
            "reconstruction error"
        ),
        "target_known_rejection_rate": (
            KNOWN_REJECTION_RATE
        ),
        "frozen_threshold": threshold,
        "actual_calibration_rejection_rate": (
            actual_rejection_rate
        ),
        "calibration_error_statistics": {
            "min": float(
                np.min(calibration_errors)
            ),
            "p50": float(
                np.percentile(
                    calibration_errors, 50
                )
            ),
            "p90": float(
                np.percentile(
                    calibration_errors, 90
                )
            ),
            "p95": float(
                np.percentile(
                    calibration_errors, 95
                )
            ),
            "p99": float(
                np.percentile(
                    calibration_errors, 99
                )
            ),
            "max": float(
                np.max(calibration_errors)
            ),
        },
        "test_data_used": False,
        "locky_data_used": False,
    }

    with OUTPUT_PATH.open("w") as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print(
        "=== Phase 8 VAE Open-set Calibration ==="
    )
    print(
        "Calibration samples:",
        len(calibration_errors),
    )
    print(
        "Target rejection rate:",
        KNOWN_REJECTION_RATE,
    )
    print(
        "Frozen reconstruction-error threshold:",
        threshold,
    )
    print(
        "Actual calibration rejection rate:",
        actual_rejection_rate,
    )
    print()
    print(
        "Locky data used: False"
    )
    print(
        "Test data used : False"
    )
    print()
    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
