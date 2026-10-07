from pathlib import Path
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

FEATURE_COLUMNS = [
    "latent_0",
    "latent_1",
    "latent_2",
    "latent_3",
    "reconstruction_error",
]


def main():
    print("=== PHASE 9 SHAP SMOKE TEST ===")
    print("SHAP:", shap.__version__)
    print("Checkpoint:", CHECKPOINT)

    assert CHECKPOINT.exists()
    assert CALIBRATION_PATH.exists()
    assert TEST_PATH.exists()

    calibration = pd.read_csv(CALIBRATION_PATH)
    test = pd.read_csv(TEST_PATH)

    X_background = calibration[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float32)

    X_test = test[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float32)

    assert np.isfinite(X_background).all()
    assert np.isfinite(X_test).all()

    # Small background set for the smoke test only.
    rng = np.random.default_rng(42)

    background_size = min(100, len(X_background))
    sample_size = min(5, len(X_test))

    background_indices = rng.choice(
        len(X_background),
        size=background_size,
        replace=False,
    )

    test_indices = rng.choice(
        len(X_test),
        size=sample_size,
        replace=False,
    )

    background = X_background[background_indices]
    samples = X_test[test_indices]

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

    # SHAP KernelExplainer is model-agnostic and suitable
    # for validating the explanation pipeline before choosing
    # the final high-performance explainer.
    explainer = shap.KernelExplainer(
        model_predict,
        background,
    )

    shap_values = explainer.shap_values(
        samples,
        nsamples=100,
    )

    if isinstance(shap_values, list):
        shap_array = np.asarray(shap_values)
    else:
        shap_array = np.asarray(shap_values)

    print("Background shape:", background.shape)
    print("Samples explained:", samples.shape)
    print("SHAP output shape:", shap_array.shape)

    assert np.isfinite(shap_array).all()

    print()
    print("SHAP smoke test: PASSED")
    print("No training performed.")
    print("No threshold tuning performed.")
    print("No model modification performed.")


if __name__ == "__main__":
    main()
