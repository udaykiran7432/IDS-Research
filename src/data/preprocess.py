from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "UGRansome.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "phase1"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "phase1"

RANDOM_STATE = 42
TEST_SIZE = 0.30

TARGET_COLUMN = "Prediction"

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
    "BTC",
    "USD",
    "Netflow_Bytes",
    "Port",
]


def main() -> None:
    print("Loading raw dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Raw shape: {df.shape}")

    # Remove incomplete rows.
    df = df.dropna().copy()

    # Remove address-like features according to the baseline methodology.
    df = df.drop(columns=DROP_COLUMNS)

    # Encode categorical input features.
    feature_encoders = {}

    for column in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        feature_encoders[column] = encoder

    # Encode target.
    target_encoder = LabelEncoder()
    df[TARGET_COLUMN] = target_encoder.fit_transform(df[TARGET_COLUMN])

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Stratified 70/30 split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Fit scaler ONLY on training numerical features.
    scaler = MinMaxScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[NUMERICAL_COLUMNS] = scaler.fit_transform(
        X_train[NUMERICAL_COLUMNS]
    )

    X_test[NUMERICAL_COLUMNS] = scaler.transform(
        X_test[NUMERICAL_COLUMNS]
    )

    # Create output directories.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Save processed datasets.
    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train.values

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test.values

    train_path = OUTPUT_DIR / "baseline_train.csv"
    test_path = OUTPUT_DIR / "baseline_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Save categorical encoders.
    for column, encoder in feature_encoders.items():
        joblib.dump(
            encoder,
            ARTIFACT_DIR / f"{column}_encoder.joblib",
        )

    # Save target encoder.
    joblib.dump(
        target_encoder,
        ARTIFACT_DIR / "target_encoder.joblib",
    )

    # Save fitted numerical scaler.
    joblib.dump(
        scaler,
        ARTIFACT_DIR / "minmax_scaler.joblib",
    )

    # Save preprocessing metadata.
    metadata = {
        "dataset": "UGRansome",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "split_strategy": "stratified",
        "dropped_columns": DROP_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numerical_columns": NUMERICAL_COLUMNS,
        "target_column": TARGET_COLUMN,
        "target_classes": {
            str(label): int(index)
            for index, label in enumerate(target_encoder.classes_)
        },
        "raw_rows_after_dropna": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "artifacts": [
            f"phase1/{column}_encoder.joblib"
            for column in CATEGORICAL_COLUMNS
        ]
        + [
            "phase1/target_encoder.joblib",
            "phase1/minmax_scaler.joblib",
        ],
    }

    metadata_path = REPORT_DIR / "baseline_preprocessing.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nBaseline preprocessing completed.")
    print(f"Training set : {train_df.shape}")
    print(f"Testing set  : {test_df.shape}")
    print(f"Train output : {train_path}")
    print(f"Test output  : {test_path}")
    print(f"Metadata     : {metadata_path}")
    print(f"Artifacts    : {ARTIFACT_DIR}")

    print("\nEncoded target classes:")
    for label, index in metadata["target_classes"].items():
        print(f"  {label} -> {index}")


if __name__ == "__main__":
    main()