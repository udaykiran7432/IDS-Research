from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "baseline_train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "baseline_test.csv"

TARGET_COLUMN = "Prediction"

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


def load_data():
    """Load the validated Phase 1 datasets for the D3QN experiment."""

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    required = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_train = set(required) - set(train_df.columns)
    missing_test = set(required) - set(test_df.columns)

    if missing_train:
        raise ValueError(
            f"Missing train columns: {sorted(missing_train)}"
        )

    if missing_test:
        raise ValueError(
            f"Missing test columns: {sorted(missing_test)}"
        )

    X_train = train_df[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32
    )
    X_test = test_df[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32
    )

    y_train_original = train_df[TARGET_COLUMN].to_numpy(
        dtype=np.int64
    )
    y_test_original = test_df[TARGET_COLUMN].to_numpy(
        dtype=np.int64
    )

    # Same binary formulation as the verified baseline:
    # original S (1) -> benign (0)
    # original A (0), SS (2) -> ransomware (1)
    y_train = (y_train_original != 1).astype(np.int64)
    y_test = (y_test_original != 1).astype(np.int64)

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        y_train_original,
        y_test_original,
    )
