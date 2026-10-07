from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_PATH = (
    PROJECT_ROOT
    / "reports/phase9/shap_representative_cases_seed_42.json"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "reports/phase9/figures"
)

FEATURE_COLUMNS = [
    "latent_0",
    "latent_1",
    "latent_2",
    "latent_3",
    "reconstruction_error",
]


def main():
    print("=== PHASE 9 SHAP FIGURES ===")

    assert REPORT_PATH.exists()

    data = json.loads(REPORT_PATH.read_text())

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated = []

    for case_name, case in data["cases"].items():
        ranking = case["shap_feature_ranking"]

        # Put features in descending absolute SHAP importance.
        ranking = sorted(
            ranking,
            key=lambda item: item["absolute_shap_value"],
        )

        features = [
            item["feature"]
            for item in ranking
        ]

        values = np.array(
            [
                item["shap_value"]
                for item in ranking
            ],
            dtype=float,
        )

        fig, ax = plt.subplots(
            figsize=(8, 5),
        )

        ax.barh(
            features,
            values,
        )

        ax.axvline(
            0,
            linewidth=1,
        )

        ax.set_xlabel(
            "SHAP value for ransomware Q-value"
        )

        ax.set_ylabel(
            "Integrated feature"
        )

        ax.set_title(
            (
                f"{case_name.replace('_', ' ').title()}\n"
                f"Family={case['family']}, "
                f"D3QN={case['dqn_prediction']}, "
                f"Open-set={case['open_set_decision']}"
            )
        )

        ax.grid(
            axis="x",
            alpha=0.2,
        )

        fig.tight_layout()

        output_path = (
            FIGURE_DIR
            / f"{case_name}_shap.png"
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        generated.append(str(output_path))

        print("Saved:", output_path)

    print()
    print(
        "Generated",
        len(generated),
        "SHAP figures."
    )


if __name__ == "__main__":
    main()
