from pathlib import Path
import json

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "UGRansome.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "phase1"
REPORT_PATH = REPORT_DIR / "dataset_profile.json"


def inspect_dataset() -> dict:
    """Inspect the raw UGRansome dataset and return a structured report."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    categorical_columns = [
        "Protcol",
        "Flag",
        "Family",
        "SeddAddress",
        "ExpAddress",
        "IPaddress",
        "Threats",
    ]

    numerical_columns = [
        "Time",
        "Clusters",
        "BTC",
        "USD",
        "Netflow_Bytes",
        "Port",
    ]

    report = {
        "dataset": {
            "name": "UGRansome",
            "path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "columns": df.columns.tolist(),
        "data_types": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "missing_values": {
            column: int(count)
            for column, count in df.isna().sum().items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "target": {
            "column": "Prediction",
            "class_counts": {
                str(label): int(count)
                for label, count in df["Prediction"].value_counts().items()
            },
            "class_percentages": {
                str(label): round(float(count / len(df) * 100), 4)
                for label, count in df["Prediction"].value_counts().items()
            },
        },
        "unique_values": {
            column: int(df[column].nunique())
            for column in df.columns
        },
        "numerical_summary": {
            column: {
                statistic: float(value)
                for statistic, value in df[column].describe().items()
            }
            for column in numerical_columns
        },
        "categorical_distributions": {
            column: {
                str(value): int(count)
                for value, count in df[column].value_counts().items()
            }
            for column in categorical_columns
        },
        "feature_target_relationships": {
            column: (
                pd.crosstab(
                    df[column],
                    df["Prediction"],
                    normalize="index",
                )
                .round(4)
                .to_dict(orient="index")
            )
            for column in categorical_columns + ["Clusters", "Port"]
        },
    }

    return report


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = inspect_dataset()

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("Dataset inspection completed.")
    print(f"Dataset : {DATA_PATH}")
    print(f"Report  : {REPORT_PATH}")
    print(f"Shape   : ({report['dataset']['rows']}, {report['dataset']['columns']})")
    print(f"Duplicates: {report['duplicate_rows']}")

    print("\nClass distribution:")
    for label, count in report["target"]["class_counts"].items():
        percentage = report["target"]["class_percentages"][label]
        print(f"  {label}: {count:,} ({percentage:.2f}%)")


if __name__ == "__main__":
    main()