import json
from pathlib import Path

import numpy as np
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports" / "phase10"

SEEDS = [42, 7, 21, 123, 2026]
EXPERIMENTS = ["e1", "e2", "e3"]

METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "fpr",
    "specificity",
]


def load_report(experiment, seed):
    path = REPORT_DIR / f"{experiment}_evaluation_seed_{seed}.json"

    with open(path) as f:
        return json.load(f)


def paired_analysis(values_a, values_b):
    """
    Compare B - A across the same seeds.

    Because n=5 is small, report both:
      - paired t-test
      - Wilcoxon signed-rank test
    without treating either as definitive evidence by itself.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)

    diff = b - a

    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))

    wins = int(np.sum(diff > 0))
    losses = int(np.sum(diff < 0))
    ties = int(np.sum(diff == 0))

    # Paired t-test
    t_stat, t_p = stats.ttest_rel(b, a)

    # Wilcoxon can fail when all differences are zero.
    try:
        w_stat, w_p = stats.wilcoxon(
            b,
            a,
            zero_method="wilcox",
            alternative="two-sided",
        )
        w_stat = float(w_stat)
        w_p = float(w_p)
    except ValueError:
        w_stat = None
        w_p = None

    # Paired Cohen's dz
    if std_diff > 0:
        cohens_dz = mean_diff / std_diff
    else:
        cohens_dz = 0.0

    return {
        "mean_difference_b_minus_a": mean_diff,
        "std_difference": std_diff,
        "wins_b": wins,
        "losses_b": losses,
        "ties": ties,
        "paired_t": {
            "statistic": float(t_stat),
            "p_value": float(t_p),
        },
        "wilcoxon": {
            "statistic": w_stat,
            "p_value": w_p,
        },
        "cohens_dz": float(cohens_dz),
        "per_seed_difference": {
            str(seed): float(d)
            for seed, d in zip(SEEDS, diff)
        },
    }


def main():
    reports = {
        exp: {
            seed: load_report(exp, seed)
            for seed in SEEDS
        }
        for exp in EXPERIMENTS
    }

    # Main known-family metrics.
    values = {
        exp: {
            metric: [
                reports[exp][seed]["known_family_classification"][metric]
                for seed in SEEDS
            ]
            for metric in METRICS
        }
        for exp in EXPERIMENTS
    }

    aggregates = {}

    for exp in EXPERIMENTS:
        aggregates[exp] = {}

        for metric in METRICS:
            arr = np.asarray(values[exp][metric], dtype=float)

            aggregates[exp][metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "per_seed": {
                    str(seed): float(value)
                    for seed, value in zip(SEEDS, arr)
                },
            }

    paired = {
        "E2_minus_E1": {},
        "E3_minus_E2": {},
    }

    for metric in METRICS:
        paired["E2_minus_E1"][metric] = paired_analysis(
            values["e1"][metric],
            values["e2"][metric],
        )

        paired["E3_minus_E2"][metric] = paired_analysis(
            values["e2"][metric],
            values["e3"][metric],
        )

    # Raw per-seed table for reproducibility.
    per_seed = {}

    for seed in SEEDS:
        per_seed[str(seed)] = {}

        for exp in EXPERIMENTS:
            per_seed[str(seed)][exp] = {
                metric: values[exp][metric][SEEDS.index(seed)]
                for metric in METRICS
            }

    result = {
        "experiment": "Phase 10.7 statistical analysis",
        "seeds": SEEDS,
        "n_seeds": len(SEEDS),
        "experiments": {
            "E1": "DQN",
            "E2": "D3QN",
            "E3": "D3QN + PER",
        },
        "evaluation_scope": "Known families only for closed-set classifier comparison; Locky remains unseen and is separately reported as closed-set forced classification.",
        "metrics": METRICS,
        "aggregates": aggregates,
        "paired_comparisons": paired,
        "per_seed": per_seed,
        "interpretation_note": (
            "Five seeds provide limited statistical power. "
            "Paired tests are reported descriptively and should not be "
            "interpreted as definitive evidence from such a small sample."
        ),
    }

    output_path = REPORT_DIR / "phase10_statistical_analysis.json"

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print("=" * 70)
    print("PHASE 10.7 — STATISTICAL ANALYSIS")
    print("=" * 70)

    print(f"Seeds: {SEEDS}")
    print()

    for exp, name in [
        ("e1", "E1 DQN"),
        ("e2", "E2 D3QN"),
        ("e3", "E3 D3QN + PER"),
    ]:
        print(f"=== {name} ===")

        for metric in METRICS:
            stats_result = aggregates[exp][metric]

            print(
                f"{metric:12s}: "
                f"{stats_result['mean']:.6f} "
                f"+/- {stats_result['std']:.6f}"
            )

        print()

    for comparison, label in [
        ("E2_minus_E1", "E2 - E1"),
        ("E3_minus_E2", "E3 - E2"),
    ]:
        print(f"=== {label} ===")

        for metric in METRICS:
            result_metric = paired[comparison][metric]

            print(
                f"{metric:12s}: "
                f"diff={result_metric['mean_difference_b_minus_a']:+.6f} "
                f"+/- {result_metric['std_difference']:.6f} | "
                f"wins={result_metric['wins_b']} "
                f"losses={result_metric['losses_b']} "
                f"ties={result_metric['ties']} | "
                f"t_p={result_metric['paired_t']['p_value']:.6f} | "
                f"wilcoxon_p={result_metric['wilcoxon']['p_value']}"
            )

        print()

    print(f"Saved: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
