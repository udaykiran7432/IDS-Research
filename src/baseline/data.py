from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "baseline_train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "baseline_test.csv"

TARGET_COLUMN = "Prediction"

# Paper baseline feature set.
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
    """Load the validated Phase 1 train/test datasets."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    missing_train = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(train_df.columns)
    missing_test = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(test_df.columns)

    if missing_train:
        raise ValueError(f"Missing train columns: {sorted(missing_train)}")

    if missing_test:
        raise ValueError(f"Missing test columns: {sorted(missing_test)}")

    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    # Preserve original 3-class labels.
    y_train_original = train_df[TARGET_COLUMN].to_numpy(dtype=np.int64)
    y_test_original = test_df[TARGET_COLUMN].to_numpy(dtype=np.int64)

    # Paper's binary formulation:
    # S (1)       -> benign (0)
    # A (0), SS(2) -> ransomware (1)
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


if __name__ == "__main__":
    (
        X_train,
        y_train,
        X_test,
        y_test,
        y_train_original,
        y_test_original,
    ) = load_data()

    print("=== Phase 2 Baseline Data Loader ===")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")

    print("\nOriginal train classes:")
    print(dict(zip(*np.unique(y_train_original, return_counts=True))))

    print("\nBinary train classes:")
    print(dict(zip(*np.unique(y_train, return_counts=True))))

    print("\nOriginal test classes:")
    print(dict(zip(*np.unique(y_test_original, return_counts=True))))

    print("\nBinary test classes:")
    print(dict(zip(*np.unique(y_test, return_counts=True))))
