"""
Phase 4: Multi-seed statistical analysis
DQN baseline vs proposed D3QN.

Seeds:
    42, 7, 21, 123, 2026

Seed 42 is the historical Phase 3 controlled experiment.
Seeds 7, 21, 123, and 2026 are Phase 4 experiments.
"""

import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE4_REPORT_DIR = PROJECT_ROOT / "reports" / "phase4"

SEEDS = [42, 7, 21, 123, 2026]

METRICS = [
    "accuracy",
    "precision",
    "recall_detection_rate",
    "f1",
    "false_positive_rate",
    "specificity",
]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall_detection_rate": "Recall / Detection Rate",
    "f1": "F1",
    "false_positive_rate": "False Positive Rate",
    "specificity": "Specificity",
}


# Historical Phase 3 Seed 42 values.
# These are taken directly from:
# reports/phase3/dqn_vs_d3qn_comparison.md
SEED_42 = {
    "dqn": {
        "accuracy": 0.955695,
        "precision": 0.931244,
        "recall_detection_rate": 0.993467,
        "f1": 0.961350,
        "false_positive_rate": 0.091343,
        "specificity": 0.908657,
    },
    "d3qn": {
        "accuracy": 0.961152,
        "precision": 0.942207,
        "recall_detection_rate": 0.990725,
        "f1": 0.965857,
        "false_positive_rate": 0.075675,
        "specificity": 0.924325,
    },
}


def load_phase4_evaluation(method: str, seed: int) -> dict:
    """Load one Phase 4 evaluation report."""
    if method == "dqn":
        path = (
            PHASE4_REPORT_DIR
            / f"dqn_seed_{seed}"
            / "baseline_evaluation.json"
        )
    else:
        path = (
            PHASE4_REPORT_DIR
            / f"d3qn_seed_{seed}"
            / "d3qn_evaluation.json"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_metrics(method: str, seed: int) -> dict:
    """Return metrics for one method and seed."""
    if seed == 42:
        return SEED_42[method]

    report = load_phase4_evaluation(method, seed)
    return report["metrics"]


def get_confusion_matrix(method: str, seed: int):
    """Return confusion matrix for Phase 4 seeds."""
    if seed == 42:
        if method == "dqn":
            return [[18095, 1819], [162, 24637]]
        return [[18407, 1507], [230, 24569]]

    report = load_phase4_evaluation(method, seed)
    return report["confusion_matrix"]


def main():
    results = {
        "dqn": {},
        "d3qn": {},
    }

    for method in ["dqn", "d3qn"]:
        for seed in SEEDS:
            results[method][seed] = get_metrics(method, seed)

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------

    aggregate = {}

    for method in ["dqn", "d3qn"]:
        aggregate[method] = {}

        for metric in METRICS:
            values = np.array(
                [results[method][seed][metric] for seed in SEEDS],
                dtype=float,
            )

            aggregate[method][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "values": values.tolist(),
            }

    # ------------------------------------------------------------------
    # Paired D3QN - DQN differences
    # ------------------------------------------------------------------

    paired_differences = {}

    for metric in METRICS:
        differences = np.array(
            [
                results["d3qn"][seed][metric]
                - results["dqn"][seed][metric]
                for seed in SEEDS
            ],
            dtype=float,
        )

        paired_differences[metric] = {
            "values": differences.tolist(),
            "mean": float(np.mean(differences)),
            "std": float(np.std(differences, ddof=1)),
            "d3qn_wins": int(np.sum(differences > 0)),
            "dqn_wins": int(np.sum(differences < 0)),
            "ties": int(np.sum(np.isclose(differences, 0.0))),
        }

    # ------------------------------------------------------------------
    # Print analysis
    # ------------------------------------------------------------------

    print("=" * 90)
    print("PHASE 4 — FIVE-SEED DQN vs D3QN ANALYSIS")
    print("=" * 90)

    print("\nSeeds:", SEEDS)
    print("Number of seeds:", len(SEEDS))

    print("\n" + "-" * 90)
    print("MEAN ± SAMPLE STANDARD DEVIATION")
    print("-" * 90)

    for metric in METRICS:
        dqn = aggregate["dqn"][metric]
        d3qn = aggregate["d3qn"][metric]

        print(
            f"{METRIC_LABELS[metric]:30s} "
            f"DQN={dqn['mean']:.6f} ± {dqn['std']:.6f} | "
            f"D3QN={d3qn['mean']:.6f} ± {d3qn['std']:.6f}"
        )

    print("\n" + "-" * 90)
    print("PAIRED D3QN − DQN DIFFERENCES")
    print("-" * 90)

    for metric in METRICS:
        diff = paired_differences[metric]

        print(
            f"{METRIC_LABELS[metric]:30s} "
            f"mean={diff['mean']:+.6f} | "
            f"std={diff['std']:.6f} | "
            f"D3QN wins={diff['d3qn_wins']} | "
            f"DQN wins={diff['dqn_wins']} | "
            f"ties={diff['ties']}"
        )

    print("\n" + "-" * 90)
    print("PER-SEED RESULTS")
    print("-" * 90)

    for seed in SEEDS:
        print(f"\nSeed {seed}")

        for metric in METRICS:
            dqn_value = results["dqn"][seed][metric]
            d3qn_value = results["d3qn"][seed][metric]
            difference = d3qn_value - dqn_value

            print(
                f"  {METRIC_LABELS[metric]:30s} "
                f"DQN={dqn_value:.6f} | "
                f"D3QN={d3qn_value:.6f} | "
                f"Δ={difference:+.6f}"
            )

    # ------------------------------------------------------------------
    # Save machine-readable aggregate analysis
    # ------------------------------------------------------------------

    analysis = {
        "experiment": "Phase 4 five-seed DQN vs D3QN analysis",
        "seeds": SEEDS,
        "number_of_seeds": len(SEEDS),
        "test_samples": 44713,
        "metrics": METRICS,
        "results": results,
        "aggregate": aggregate,
        "paired_differences": paired_differences,
        "confusion_matrices": {
            method: {
                str(seed): get_confusion_matrix(method, seed)
                for seed in SEEDS
            }
            for method in ["dqn", "d3qn"]
        },
    }

    output_json = PHASE4_REPORT_DIR / "phase4_five_seed_analysis.json"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    print("\nSaved:")
    print(output_json)


if __name__ == "__main__":
    main()