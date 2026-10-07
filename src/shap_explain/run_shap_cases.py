from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch
import shap


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
    / "reports/phase9/shap_representative_cases_seed_42.json"
)

FEATURE_COLUMNS = [
    "latent_0",
    "latent_1",
    "latent_2",
    "latent_3",
    "reconstruction_error",
]

THRESHOLD = 0.23906713426113124

RANDOM_SEED = 42
BACKGROUND_SIZE = 100
NSAMPLES = 100


def main():
    print("=== PHASE 9 SHAP REPRESENTATIVE CASES ===")
    print("Checkpoint:", CHECKPOINT)
    print("Frozen VAE threshold:", THRESHOLD)
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
    assert len(metadata) == 62257

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
    # Load frozen Phase 8 model
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
    # Compute all test predictions once
    # ------------------------------------------------------------

    q_values = model_predict(X_test)
    dqn_predictions = np.argmax(q_values, axis=1)

    reconstruction_error = X_test[:, 4]

    unknown_mask = reconstruction_error >= THRESHOLD

    final_predictions = dqn_predictions.copy()
    final_predictions[unknown_mask] = -1

    # ------------------------------------------------------------
    # Select representative cases
    # ------------------------------------------------------------

    correct_benign = np.where(
        (families != "Locky")
        & (y_test == 0)
        & (dqn_predictions == 0)
        & (~unknown_mask)
    )[0]

    correct_ransomware = np.where(
        (families != "Locky")
        & (y_test == 1)
        & (dqn_predictions == 1)
        & (~unknown_mask)
    )[0]

    correct_locky_unknown = np.where(
        (families == "Locky")
        & unknown_mask
    )[0]

    locky_false_accept = np.where(
        (families == "Locky")
        & (~unknown_mask)
    )[0]

    known_misclassification = np.where(
        (families != "Locky")
        & (dqn_predictions != y_test)
        & (~unknown_mask)
    )[0]

    candidate_sets = {
        "known_benign_correct": correct_benign,
        "known_ransomware_correct": correct_ransomware,
        "locky_correctly_unknown": correct_locky_unknown,
        "locky_false_accept": locky_false_accept,
        "known_misclassification": known_misclassification,
    }

    for name, indices in candidate_sets.items():
        if len(indices) == 0:
            raise RuntimeError(
                f"No representative sample available for: {name}"
            )

    # Deterministic representative selection.
    #
    # For each case we select the sample closest to the median
    # reconstruction error of its candidate set. This avoids
    # selecting an extreme outlier merely because it is easy to find.
    selected = {}

    for name, indices in candidate_sets.items():
        values = reconstruction_error[indices]
        median_value = np.median(values)

        selected_index = indices[
            np.argmin(np.abs(values - median_value))
        ]

        selected[name] = int(selected_index)

    print("Selected representative indices:")
    for name, index in selected.items():
        print(
            f"  {name}: index={index}, "
            f"family={families[index]}, "
            f"true={y_test[index]}, "
            f"dqn={dqn_predictions[index]}, "
            f"unknown={bool(unknown_mask[index])}, "
            f"reconstruction_error={reconstruction_error[index]:.6f}"
        )

    print()

    # ------------------------------------------------------------
    # SHAP background
    # ------------------------------------------------------------

    rng = np.random.default_rng(RANDOM_SEED)

    background_size = min(
        BACKGROUND_SIZE,
        len(X_background),
    )

    background_indices = rng.choice(
        len(X_background),
        size=background_size,
        replace=False,
    )

    background = X_background[background_indices]

    # ------------------------------------------------------------
    # Kernel SHAP
    # ------------------------------------------------------------

    explainer = shap.KernelExplainer(
        model_predict,
        background,
    )

    sample_indices = np.array(
        list(selected.values()),
        dtype=np.int64,
    )

    samples = X_test[sample_indices]

    print(
        f"Computing SHAP for {len(samples)} "
        f"representative cases..."
    )

    shap_values = explainer.shap_values(
        samples,
        nsamples=NSAMPLES,
    )

    shap_array = np.asarray(shap_values)

    # SHAP 0.52 returns:
    # samples x features x outputs
    #
    # We explain output/action 1:
    # ransomware Q-value.
    if shap_array.ndim == 3:
        ransomware_shap = shap_array[:, :, 1]
    elif shap_array.ndim == 2:
        ransomware_shap = shap_array
    else:
        raise RuntimeError(
            f"Unexpected SHAP shape: {shap_array.shape}"
        )

    assert ransomware_shap.shape == (
        len(sample_indices),
        len(FEATURE_COLUMNS),
    )

    assert np.isfinite(ransomware_shap).all()

    # ------------------------------------------------------------
    # Build report
    # ------------------------------------------------------------

    cases = {}

    selected_items = list(selected.items())

    for row, (case_name, index) in enumerate(selected_items):
        feature_values = X_test[index]
        feature_shap = ransomware_shap[row]

        ranking = sorted(
            zip(
                FEATURE_COLUMNS,
                feature_values,
                feature_shap,
            ),
            key=lambda item: abs(item[2]),
            reverse=True,
        )

        cases[case_name] = {
            "test_index": int(index),
            "family": str(families[index]),
            "true_binary_label": int(y_test[index]),
            "dqn_prediction": int(dqn_predictions[index]),
            "final_prediction": int(final_predictions[index]),
            "open_set_decision": (
                "UNKNOWN"
                if unknown_mask[index]
                else "KNOWN"
            ),
            "q_value_benign": float(q_values[index, 0]),
            "q_value_ransomware": float(q_values[index, 1]),
            "q_margin": float(
                abs(q_values[index, 1] - q_values[index, 0])
            ),
            "reconstruction_error": float(
                reconstruction_error[index]
            ),
            "threshold": THRESHOLD,
            "features": {
                feature: float(value)
                for feature, value in zip(
                    FEATURE_COLUMNS,
                    feature_values,
                )
            },
            "shap_ransomware_q_value": {
                feature: float(value)
                for feature, value in zip(
                    FEATURE_COLUMNS,
                    feature_shap,
                )
            },
            "shap_feature_ranking": [
                {
                    "feature": feature,
                    "value": float(value),
                    "shap_value": float(shap_value),
                    "absolute_shap_value": float(
                        abs(shap_value)
                    ),
                }
                for feature, value, shap_value in ranking
            ],
        }

    report = {
        "experiment": "Phase 9 SHAP representative cases",
        "seed": RANDOM_SEED,
        "shap_version": shap.__version__,
        "checkpoint": str(CHECKPOINT),
        "feature_columns": FEATURE_COLUMNS,
        "background_source": "Phase 8 calibration representations",
        "background_size": int(background_size),
        "nsamples": NSAMPLES,
        "explained_output": "D3QN ransomware Q-value (action 1)",
        "open_set_signal": "VAE reconstruction error",
        "frozen_threshold": THRESHOLD,
        "test_threshold_tuning": False,
        "model_retrained": False,
        "cases": cases,
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
    print("SHAP representative-case experiment: PASSED")
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
