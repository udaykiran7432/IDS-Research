from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "phase10"
FIG_DIR = REPORT_DIR / "figures"

SEEDS = [42, 7, 21, 123, 2026]
EXPERIMENTS = {
    "E1 DQN": "e1",
    "E2 D3QN": "e2",
    "E3 D3QN + PER": "e3",
}

METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "fpr",
]

DISPLAY = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "fpr": "FPR",
}


def load_evaluation(exp_code: str, seed: int) -> dict:
    path = REPORT_DIR / f"{exp_code}_evaluation_seed_{seed}.json"
    with path.open() as f:
        return json.load(f)


def get_metrics(payload: dict) -> dict:
    return payload["known_family_classification"]


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for exp_name, exp_code in EXPERIMENTS.items():
        for seed in SEEDS:
            payload = load_evaluation(exp_code, seed)
            metrics = get_metrics(payload)

            row = {
                "experiment": exp_name,
                "seed": seed,
            }

            for metric in METRICS:
                row[metric] = metrics[metric]

            rows.append(row)

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------
    summary_rows = []

    for exp_name in EXPERIMENTS:
        subset = df[df["experiment"] == exp_name]

        row = {"experiment": exp_name}

        for metric in METRICS:
            values = subset[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)
            row[f"{metric}_min"] = values.min()
            row[f"{metric}_max"] = values.max()

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        REPORT_DIR / "phase10_summary_table.csv",
        index=False,
        float_format="%.8f",
    )

    # ------------------------------------------------------------
    # Paired differences
    # ------------------------------------------------------------
    paired_rows = []

    for seed in SEEDS:
        e1 = df[(df.experiment == "E1 DQN") & (df.seed == seed)].iloc[0]
        e2 = df[(df.experiment == "E2 D3QN") & (df.seed == seed)].iloc[0]
        e3 = df[(df.experiment == "E3 D3QN + PER") & (df.seed == seed)].iloc[0]

        for comparison, a, b in [
            ("E2 D3QN - E1 DQN", e1, e2),
            ("E3 D3QN + PER - E2 D3QN", e2, e3),
        ]:
            row = {
                "comparison": comparison,
                "seed": seed,
            }

            for metric in METRICS:
                row[metric] = b[metric] - a[metric]

            paired_rows.append(row)

    paired = pd.DataFrame(paired_rows)

    paired.to_csv(
        REPORT_DIR / "phase10_paired_differences.csv",
        index=False,
        float_format="%.8f",
    )

    # ------------------------------------------------------------
    # Markdown results table
    # ------------------------------------------------------------
    lines = [
        "# Phase 10 Results",
        "",
        "Controlled family-held-out evaluation across five seeds.",
        "",
        "Values are mean ± sample standard deviation.",
        "",
        "| Experiment | Accuracy | Precision | Recall | F1 | FPR |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in summary.iterrows():
        cells = [row["experiment"]]

        for metric in METRICS:
            cells.append(
                f"{row[f'{metric}_mean']:.4f} ± "
                f"{row[f'{metric}_std']:.4f}"
            )

        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Paired Differences",
            "",
            "| Comparison | Metric | Mean Difference | SD |",
            "|---|---|---:|---:|",
        ]
    )

    for comparison in paired["comparison"].unique():
        subset = paired[paired["comparison"] == comparison]

        for metric in METRICS:
            values = subset[metric].to_numpy(dtype=float)
            lines.append(
                f"| {comparison} | {DISPLAY[metric]} | "
                f"{values.mean():+.6f} | {values.std(ddof=1):.6f} |"
            )

    (REPORT_DIR / "phase10_results_table.md").write_text(
        "\n".join(lines) + "\n"
    )

    # ------------------------------------------------------------
    # Figure 1: mean ± SD comparison
    # ------------------------------------------------------------
    x = np.arange(len(METRICS))
    width = 0.25

    plt.figure(figsize=(11, 6))

    for i, exp_name in enumerate(EXPERIMENTS):
        subset = summary[summary["experiment"] == exp_name].iloc[0]

        means = [subset[f"{m}_mean"] for m in METRICS]
        stds = [subset[f"{m}_std"] for m in METRICS]

        plt.errorbar(
            x + (i - 1) * width,
            means,
            yerr=stds,
            fmt="o",
            capsize=4,
            label=exp_name,
        )

    plt.xticks(x, [DISPLAY[m] for m in METRICS])
    plt.ylabel("Score")
    plt.title("Phase 10 Controlled Evaluation: Mean ± SD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "phase10_mean_sd_comparison.png", dpi=200)
    plt.close()

    # ------------------------------------------------------------
    # Figure 2: F1 across seeds
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    for exp_name in EXPERIMENTS:
        subset = df[df["experiment"] == exp_name].sort_values("seed")
        plt.plot(
            subset["seed"],
            subset["f1"],
            marker="o",
            label=exp_name,
        )

    plt.xlabel("Seed")
    plt.ylabel("F1")
    plt.title("F1 Across Random Seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "phase10_f1_across_seeds.png", dpi=200)
    plt.close()

    # ------------------------------------------------------------
    # Figure 3: D3QN - DQN
    # ------------------------------------------------------------
    subset = paired[paired["comparison"] == "E2 D3QN - E1 DQN"]

    plt.figure(figsize=(11, 6))

    for metric in METRICS:
        plt.plot(
            subset["seed"],
            subset[metric],
            marker="o",
            label=DISPLAY[metric],
        )

    plt.axhline(0, linewidth=1)
    plt.xlabel("Seed")
    plt.ylabel("Paired Difference")
    plt.title("D3QN − DQN Across Seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "phase10_d3qn_minus_dqn.png", dpi=200)
    plt.close()

    # ------------------------------------------------------------
    # Figure 4: PER - D3QN
    # ------------------------------------------------------------
    subset = paired[
        paired["comparison"] == "E3 D3QN + PER - E2 D3QN"
    ]

    plt.figure(figsize=(11, 6))

    for metric in METRICS:
        plt.plot(
            subset["seed"],
            subset[metric],
            marker="o",
            label=DISPLAY[metric],
        )

    plt.axhline(0, linewidth=1)
    plt.xlabel("Seed")
    plt.ylabel("Paired Difference")
    plt.title("D3QN + PER − D3QN Across Seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "phase10_per_minus_d3qn.png", dpi=200)
    plt.close()

    # ------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------
    manifest = {
        "phase": 10,
        "description": "Controlled evaluation outputs generated from saved experiment JSON reports.",
        "seeds": SEEDS,
        "experiments": list(EXPERIMENTS.keys()),
        "metrics": METRICS,
        "source_reports": [
            f"{code}_evaluation_seed_{seed}.json"
            for code in EXPERIMENTS.values()
            for seed in SEEDS
        ],
        "outputs": [
            "phase10_summary_table.csv",
            "phase10_paired_differences.csv",
            "phase10_results_table.md",
            "figures/phase10_mean_sd_comparison.png",
            "figures/phase10_f1_across_seeds.png",
            "figures/phase10_d3qn_minus_dqn.png",
            "figures/phase10_per_minus_d3qn.png",
        ],
    }

    with (REPORT_DIR / "phase10_output_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print("Phase 10 tables, figures, and manifest generated.")
    print(f"Output directory: {REPORT_DIR}")


if __name__ == "__main__":
    main()
