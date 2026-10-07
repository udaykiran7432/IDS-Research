from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "UGRansome.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "phase7_open_set"
REPORT_DIR = PROJECT_ROOT / "reports" / "phase7"

FAMILY_COLUMN = "Family"
TARGET_COLUMN = "Prediction"

UNSEEN_FAMILY_NAME = "Locky"

RANDOM_STATE = 42
TEST_SIZE = 0.30
CALIBRATION_SIZE = 0.20

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

DROP_COLUMNS = [
    "SeddAddress",
    "ExpAddress",
    "IPaddress",
]

CATEGORICAL_COLUMNS = [
    "Protcol",
    "Flag",
    "Family",
    "Threats",
]

NUMERICAL_COLUMNS = [
    "Time",
    "Clusters",
    "USD",
    "BTC",
]


def binary_target(values):
    """
    Original UGRansome target:
        S  -> benign (0)
        A  -> ransomware (1)
        SS -> ransomware (1)
    """
    return (values != "S").astype(np.int64)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw UGRansome dataset...")
    df = pd.read_csv(RAW_PATH)

    print("Raw shape:", df.shape)

    required_columns = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
        + DROP_COLUMNS
    )

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    # ---------------------------------------------------------
    # 1. Identify the unseen family
    # ---------------------------------------------------------
    family_values = df[FAMILY_COLUMN].astype(str)

    if UNSEEN_FAMILY_NAME not in set(family_values.unique()):
        raise RuntimeError(
            f"Unseen family '{UNSEEN_FAMILY_NAME}' was not found."
        )

    unseen_mask = family_values == UNSEEN_FAMILY_NAME

    unseen_df = df.loc[unseen_mask].copy()
    known_df = df.loc[~unseen_mask].copy()

    print()
    print("Unseen family:", UNSEEN_FAMILY_NAME)
    print("Unseen samples:", len(unseen_df))
    print("Known-family samples:", len(known_df))

    # ---------------------------------------------------------
    # 2. Split ONLY known families into train/test
    # ---------------------------------------------------------
    known_train_raw, known_test_raw = train_test_split(
        known_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=known_df[TARGET_COLUMN].astype(str),
    )

    # All unseen-family samples are final-test-only.
    phase7_test_raw = pd.concat(
        [known_test_raw, unseen_df],
        axis=0,
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # 3. Split known training data into model train/calibration
    # ---------------------------------------------------------
    calibration_strata = (
        known_train_raw[FAMILY_COLUMN].astype(str)
        + "_"
        + known_train_raw[TARGET_COLUMN].astype(str)
    )

    phase7_train_raw, calibration_raw = train_test_split(
        known_train_raw,
        test_size=CALIBRATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=calibration_strata,
    )

    # ---------------------------------------------------------
    # 4. Hard leakage checks
    # ---------------------------------------------------------
    for name, split in [
        ("train", phase7_train_raw),
        ("calibration", calibration_raw),
    ]:
        count = int(
            (split[FAMILY_COLUMN].astype(str) == UNSEEN_FAMILY_NAME).sum()
        )

        if count != 0:
            raise RuntimeError(
                f"{name} contains {count} unseen-family samples."
            )

    unseen_test_count = int(
        (
            phase7_test_raw[FAMILY_COLUMN].astype(str)
            == UNSEEN_FAMILY_NAME
        ).sum()
    )

    if unseen_test_count != len(unseen_df):
        raise RuntimeError(
            "Not all unseen-family samples reached the final test set."
        )

    # ---------------------------------------------------------
    # 5. Fit preprocessing ONLY on phase7 training data
    # ---------------------------------------------------------
    train_features = phase7_train_raw[FEATURE_COLUMNS].copy()
    calibration_features = calibration_raw[FEATURE_COLUMNS].copy()
    test_features = phase7_test_raw[FEATURE_COLUMNS].copy()

    # Categorical preprocessing:
    # unknown categories are represented as -1.
    categorical_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.float64,
    )

    train_features[CATEGORICAL_COLUMNS] = (
        categorical_encoder.fit_transform(
            train_features[CATEGORICAL_COLUMNS]
        )
    )

    calibration_features[CATEGORICAL_COLUMNS] = (
        categorical_encoder.transform(
            calibration_features[CATEGORICAL_COLUMNS]
        )
    )

    test_features[CATEGORICAL_COLUMNS] = (
        categorical_encoder.transform(
            test_features[CATEGORICAL_COLUMNS]
        )
    )

    # Numerical preprocessing:
    # scaler is fitted ONLY on phase7 training data.
    scaler = MinMaxScaler()

    train_features[NUMERICAL_COLUMNS] = scaler.fit_transform(
        train_features[NUMERICAL_COLUMNS]
    )

    calibration_features[NUMERICAL_COLUMNS] = scaler.transform(
        calibration_features[NUMERICAL_COLUMNS]
    )

    test_features[NUMERICAL_COLUMNS] = scaler.transform(
        test_features[NUMERICAL_COLUMNS]
    )

    # Ensure canonical feature order.
    train_features = train_features[FEATURE_COLUMNS]
    calibration_features = calibration_features[FEATURE_COLUMNS]
    test_features = test_features[FEATURE_COLUMNS]

    # ---------------------------------------------------------
    # 6. Add binary target
    # ---------------------------------------------------------
    train_output = train_features.copy()
    calibration_output = calibration_features.copy()
    test_output = test_features.copy()

    train_output[TARGET_COLUMN] = binary_target(
        phase7_train_raw[TARGET_COLUMN].astype(str).values
    )

    calibration_output[TARGET_COLUMN] = binary_target(
        calibration_raw[TARGET_COLUMN].astype(str).values
    )

    test_output[TARGET_COLUMN] = binary_target(
        phase7_test_raw[TARGET_COLUMN].astype(str).values
    )

    # ---------------------------------------------------------
    # 7. Save datasets
    # ---------------------------------------------------------
    train_path = OUTPUT_DIR / "train.csv"
    calibration_path = OUTPUT_DIR / "calibration.csv"
    test_path = OUTPUT_DIR / "test.csv"

    train_output.to_csv(train_path, index=False)
    calibration_output.to_csv(calibration_path, index=False)
    test_output.to_csv(test_path, index=False)

    # Save family information separately for evaluation.
    test_metadata = phase7_test_raw[
        [FAMILY_COLUMN, TARGET_COLUMN]
    ].copy()

    test_metadata_path = OUTPUT_DIR / "test_metadata.csv"
    test_metadata.to_csv(test_metadata_path, index=False)

    # ---------------------------------------------------------
    # 8. Metadata/report
    # ---------------------------------------------------------
    def family_counts(split):
        return {
            str(k): int(v)
            for k, v in split[FAMILY_COLUMN]
            .astype(str)
            .value_counts()
            .sort_index()
            .items()
        }

    def target_counts(split):
        return {
            str(k): int(v)
            for k, v in split[TARGET_COLUMN]
            .astype(str)
            .value_counts()
            .sort_index()
            .items()
        }

    metadata = {
        "experiment": "Phase 7 family-held-out open-set split",
        "random_state": RANDOM_STATE,
        "unseen_family": UNSEEN_FAMILY_NAME,
        "unseen_family_source_samples": int(len(unseen_df)),
        "test_size_known_families": TEST_SIZE,
        "calibration_size_from_known_training": CALIBRATION_SIZE,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numerical_columns": NUMERICAL_COLUMNS,
        "preprocessing": {
            "categorical": (
                "OrdinalEncoder fitted only on Phase 7 training data; "
                "unknown categories mapped to -1."
            ),
            "numerical": (
                "MinMaxScaler fitted only on Phase 7 training data."
            ),
        },
        "splits": {
            "train": {
                "samples": int(len(phase7_train_raw)),
                "unseen_family_samples": int(
                    (
                        phase7_train_raw[FAMILY_COLUMN].astype(str)
                        == UNSEEN_FAMILY_NAME
                    ).sum()
                ),
                "families": family_counts(phase7_train_raw),
                "targets": target_counts(phase7_train_raw),
            },
            "calibration": {
                "samples": int(len(calibration_raw)),
                "unseen_family_samples": int(
                    (
                        calibration_raw[FAMILY_COLUMN].astype(str)
                        == UNSEEN_FAMILY_NAME
                    ).sum()
                ),
                "families": family_counts(calibration_raw),
                "targets": target_counts(calibration_raw),
            },
            "test": {
                "samples": int(len(phase7_test_raw)),
                "unseen_family_samples": int(unseen_test_count),
                "known_family_samples": int(
                    len(phase7_test_raw) - unseen_test_count
                ),
                "families": family_counts(phase7_test_raw),
                "targets": target_counts(phase7_test_raw),
            },
        },
        "checks": {
            "unseen_absent_from_train": True,
            "unseen_absent_from_calibration": True,
            "unseen_present_in_test": True,
            "preprocessing_fitted_only_on_train": True,
        },
    }

    metadata_path = REPORT_DIR / "phase7_split.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # ---------------------------------------------------------
    # 9. Final console summary
    # ---------------------------------------------------------
    print()
    print("PHASE 7 SPLIT CREATED")
    print("=====================")
    print("Train samples:", len(phase7_train_raw))
    print("Calibration samples:", len(calibration_raw))
    print("Test samples:", len(phase7_test_raw))
    print()
    print("Unseen family:", UNSEEN_FAMILY_NAME)
    print("Training unseen samples:", 0)
    print("Calibration unseen samples:", 0)
    print("Test unseen samples:", unseen_test_count)
    print()
    print("Train families:")
    print(metadata["splits"]["train"]["families"])
    print()
    print("Test families:")
    print(metadata["splits"]["test"]["families"])
    print()
    print("Saved:")
    print(train_path)
    print(calibration_path)
    print(test_path)
    print(test_metadata_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
