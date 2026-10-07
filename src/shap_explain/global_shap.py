from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROPOSED_DIR = PROJECT_ROOT / "src" / "proposed"
if str(PROPOSED_DIR) not in sys.path:
    sys.path.insert(0, str(PROPOSED_DIR))

from dqn import D3QN


CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts/phase8/d3qn_per_vae_seed_42.pt"
)

CALIBRATION_PATH = (
    PROJECT_ROOT
    / "data/processed/phase8_integrated/calibration.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data/processed/phase8_integrated/test.csv"
)

META_PATH = (
    PROJECT_ROOT
    / "data/processed/phase7_open_set/test_metadata.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports/phase9/global_shap_seed_42.json"
)

FIGURE_PATH = (
    PROJECT_ROOT
    / "reports/phase9/figures/global_shap_importance.png"
)

FEATURE_COLUMNS = [
    "latent_0",
    "latent_1",
    "latent_2",
    "latent_3",
    "reconstruction_error",
]

RANDOM_SEED = 42

BACKGROUND_SIZE = 100

# Larger than the representative-case experiment,
# but deliberately bounded because Kernel SHAP is expensive.
TOTAL_EXPLANATION_SAMPLES = 100

NSAMPLES = 100

THRESHOLD = 0.23906713426113124


def main():
    print("=== PHASE 9 GLOBAL SHAP ANALYSIS ===")
    print("SHAP:", shap.__version__)
    print("Background:", BACKGROUND_SIZE)
    print("Explanation samples:", TOTAL_EXPLANATION_SAMPLES)
    print("Kernel SHAP nsamples:", NSAMPLES)
    print()

    assert CHECKPOINT.exists()
    assert CALIBRATION_PATH.exists()
    assert TEST_PATH.exists()
    assert META_PATH.exists()

    calibration = pd.read_csv(CALIBRATION_PATH)
    test = pd.read_csv(TEST_PATH)
    metadata = pd.read_csv(META_PATH)

    assert len(test) == len(metadata)
    assert len(test) == 62257

    X_background = calibration[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float32)

    X_test = test[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float32)

    y_test = test["Prediction"].to_numpy(dtype=np.int64)

    families = metadata["Family"].astype(str).to_numpy()

    assert np.isfinite(X_background).all()
    assert np.isfinite(X_test).all()

    # ------------------------------------------------------------
    # Frozen Phase 8 model
    # ------------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint["state_dim"] == 5
    assert checkpoint["action_dim"] == 2
    assert checkpoint["feature_columns"] == FEATURE_COLUMNS

    model = D3QN(
        state_dim=5,
        action_dim=2,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    def model_predict(x):
        x = np.asarray(x, dtype=np.float32)

        with torch.no_grad():
            tensor = torch.tensor(
                x,
                dtype=torch.float32,
            )

            q_values = model(tensor)

        return q_values.cpu().numpy()

    # ------------------------------------------------------------
    # Background
    # ------------------------------------------------------------

    rng = np.random.default_rng(RANDOM_SEED)

    background_indices = rng.choice(
        len(X_background),
        size=min(BACKGROUND_SIZE, len(X_background)),
        replace=False,
    )

    background = X_background[background_indices]

    # ------------------------------------------------------------
    # Controlled test sample
    #
    # Equal allocation:
    #   25 known
    #   25 Locky
    #   25 known benign
    #   25 known ransomware
    #
    # This prevents the large Locky group from dominating
    # the global explanation.
    # ------------------------------------------------------------

    known_mask = families != "Locky"
    locky_mask = families == "Locky"

    benign_mask = known_mask & (y_test == 0)
    ransomware_mask = known_mask & (y_test == 1)

    per_group = TOTAL_EXPLANATION_SAMPLES // 4

    groups = {
        "known_benign": np.where(benign_mask)[0],
        "known_ransomware": np.where(ransomware_mask)[0],
        "locky": np.where(locky_mask)[0],
    }

    selected_groups = {}

    # Three groups alone would give 99 samples if equally divided.
    # We therefore allocate 25/25/25 and use the remaining 25
    # as a second balanced known-vs-Locky allocation.
    base_per_group = 25

    for name, indices in groups.items():
        selected_groups[name] = rng.choice(
            indices,
            size=base_per_group,
            replace=False,
        )

    remaining_needed = (
        TOTAL_EXPLANATION_SAMPLES
        - sum(len(v) for v in selected_groups.values())
    )

    remaining_pool = np.concatenate(
        [
            np.where(known_mask)[0],
            np.where(locky_mask)[0],
        ]
    )

    extra_indices = rng.choice(
        remaining_pool,
        size=remaining_needed,
        replace=False,
    )

    sample_indices = np.concatenate(
        [
            selected_groups["known_benign"],
            selected_groups["known_ransomware"],
            selected_groups["locky"],
            extra_indices,
        ]
    )

    rng.shuffle(sample_indices)

    samples = X_test[sample_indices]

    print("Selected explanation samples:", len(samples))
    print(
        "Known:",
        int(np.sum(families[sample_indices] != "Locky")),
    )
    print(
        "Locky:",
        int(np.sum(families[sample_indices] == "Locky")),
    )
    print()

    # ------------------------------------------------------------
    # Predictions / open-set status
    # ------------------------------------------------------------

    q_values = model_predict(samples)

    dqn_predictions = np.argmax(
        q_values,
        axis=1,
    )

    reconstruction_error = samples[:, 4]

    unknown = reconstruction_error >= THRESHOLD

    # ------------------------------------------------------------
    # Kernel SHAP
    # ------------------------------------------------------------

    explainer = shap.KernelExplainer(
        model_predict,
        background,
    )

    print("Computing Kernel SHAP...")

    shap_values = explainer.shap_values(
        samples,
        nsamples=NSAMPLES,
    )

    shap_array = np.asarray(shap_values)

    if shap_array.ndim == 3:
        # samples × features × classes
        ransomware_shap = shap_array[:, :, 1]
    elif shap_array.ndim == 2:
        ransomware_shap = shap_array
    else:
        raise RuntimeError(
            f"Unexpected SHAP output shape: {shap_array.shape}"
        )

    assert ransomware_shap.shape == (
        len(samples),
        len(FEATURE_COLUMNS),
    )

    assert np.isfinite(ransomware_shap).all()

    # ------------------------------------------------------------
    # Global importance
    # ------------------------------------------------------------

    mean_abs_all = np.mean(
        np.abs(ransomware_shap),
        axis=0,
    )

    order = np.argsort(
        mean_abs_all
    )[::-1]

    global_importance = []

    for index in order:
        global_importance.append(
            {
                "feature": FEATURE_COLUMNS[index],
                "mean_absolute_shap": float(
                    mean_abs_all[index]
                ),
            }
        )

    # ------------------------------------------------------------
    # Group-specific importance
    # ------------------------------------------------------------

    group_masks = {
        "known_benign": (
            families[sample_indices] != "Locky"
        ) & (
            y_test[sample_indices] == 0
        ),
        "known_ransomware": (
            families[sample_indices] != "Locky"
        ) & (
            y_test[sample_indices] == 1
        ),
        "locky": (
            families[sample_indices] == "Locky"
        ),
        "unknown_decisions": unknown,
        "known_decisions": ~unknown,
    }

    group_importance = {}

    for group_name, mask in group_masks.items():
        count = int(np.sum(mask))

        if count == 0:
            group_importance[group_name] = {
                "count": 0,
                "features": [],
            }
            continue

        values = np.mean(
            np.abs(
                ransomware_shap[mask]
            ),
            axis=0,
        )

        group_order = np.argsort(
            values
        )[::-1]

        group_importance[group_name] = {
            "count": count,
            "features": [
                {
                    "feature": FEATURE_COLUMNS[index],
                    "mean_absolute_shap": float(
                        values[index]
                    ),
                }
                for index in group_order
            ],
        }

    # ------------------------------------------------------------
    # Directional statistics
    # ------------------------------------------------------------

    mean_signed_shap = np.mean(
        ransomware_shap,
        axis=0,
    )

    directional = [
        {
            "feature": FEATURE_COLUMNS[index],
            "mean_signed_shap": float(
                mean_signed_shap[index]
            ),
            "positive_fraction": float(
                np.mean(
                    ransomware_shap[:, index] > 0
                )
            ),
            "negative_fraction": float(
                np.mean(
                    ransomware_shap[:, index] < 0
                )
            ),
        }
        for index in range(len(FEATURE_COLUMNS))
    ]

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------

    FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features = [
        item["feature"]
        for item in global_importance
    ]

    values = [
        item["mean_absolute_shap"]
        for item in global_importance
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5),
    )

    ax.barh(
        features[::-1],
        values[::-1],
    )

    ax.set_xlabel(
        "Mean absolute SHAP value"
    )

    ax.set_ylabel(
        "Integrated feature"
    )

    ax.set_title(
        "Global SHAP Importance — D3QN Ransomware Q-value"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------

    report = {
        "experiment": "Phase 9 global SHAP analysis",
        "seed": RANDOM_SEED,
        "shap_version": shap.__version__,
        "checkpoint": str(CHECKPOINT),
        "feature_columns": FEATURE_COLUMNS,
        "background_size": int(len(background)),
        "explanation_samples": int(len(samples)),
        "nsamples": NSAMPLES,
        "explained_output": (
            "D3QN ransomware Q-value (action 1)"
        ),
        "open_set_signal": (
            "VAE reconstruction error"
        ),
        "frozen_threshold": THRESHOLD,
        "threshold_tuning_performed": False,
        "model_retrained": False,
        "global_mean_absolute_shap": global_importance,
        "group_mean_absolute_shap": group_importance,
        "directional_statistics": directional,
        "figure": str(FIGURE_PATH),
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print("=== GLOBAL SHAP IMPORTANCE ===")

    for item in global_importance:
        print(
            f'{item["feature"]:24s} '
            f'{item["mean_absolute_shap"]:.6f}'
        )

    print()
    print(
        "Saved report:",
        REPORT_PATH,
    )

    print(
        "Saved figure:",
        FIGURE_PATH,
    )


if __name__ == "__main__":
    main()
