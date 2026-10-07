import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Reuse the validated Phase 7 model implementation.
sys.path.insert(0, str(PROJECT_ROOT / "src" / "proposed"))

from agent import D3QNAgent


SEED = 42
DEVICE = "cpu"

CALIBRATION_PATH = PROJECT_ROOT / "data/processed/phase7_open_set/calibration.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts/phase7/d3qn_per_open_set_seed_42.pt"
OUTPUT_PATH = PROJECT_ROOT / "reports/phase7/score_comparison_seed_42.json"

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

# Frozen operating point chosen previously.
REJECTION_RATE = 0.05


def load_model():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    agent = D3QNAgent(
        state_dim=len(FEATURE_COLUMNS),
        action_dim=2,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=0.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        batch_size=64,
        seed=SEED,
        device=DEVICE,
        use_per=True,
    )

    # Phase 7 training checkpoint stores the online D3QN weights
    # under "model_state_dict".
    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Expected Phase 7 checkpoint key 'model_state_dict'. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    state_dict = checkpoint["model_state_dict"]

    agent.policy_net.load_state_dict(state_dict)
    agent.policy_net.eval()

    return agent


def get_q_values(agent, X, batch_size=4096):
    outputs = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.tensor(
                X[start:start + batch_size],
                dtype=torch.float32,
                device=DEVICE,
            )
            q = agent.policy_net(batch).detach().cpu().numpy()
            outputs.append(q)

    return np.concatenate(outputs, axis=0)


def percentile_threshold_for_rejection(score, rejection_rate):
    """
    Higher score = more uncertain / more likely to reject.

    Reject the highest rejection_rate fraction.
    """
    return float(np.quantile(score, 1.0 - rejection_rate))


def evaluate_score(score, errors, threshold):
    rejected = score >= threshold

    rejected_count = int(rejected.sum())
    error_count = int(errors.sum())

    incorrect_captured = int((rejected & errors).sum())

    correct = ~errors

    correct_rejected = int((rejected & correct).sum())

    remaining = ~rejected
    remaining_errors = int((remaining & errors).sum())

    remaining_count = int(remaining.sum())
    remaining_accuracy = (
        float((remaining_count - remaining_errors) / remaining_count)
        if remaining_count > 0
        else None
    )

    return {
        "threshold": threshold,
        "rejected_count": rejected_count,
        "rejection_rate": float(rejected.mean()),
        "incorrect_predictions": error_count,
        "incorrect_predictions_captured": incorrect_captured,
        "incorrect_capture_rate": (
            float(incorrect_captured / error_count)
            if error_count > 0
            else None
        ),
        "correct_predictions_rejected": correct_rejected,
        "remaining_count": remaining_count,
        "remaining_accuracy": remaining_accuracy,
    }


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    df = pd.read_csv(CALIBRATION_PATH)

    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df["Prediction"].to_numpy(dtype=np.int64)

    # The calibration set contains only known families.
    if (df["Family"] == -1).any():
        raise RuntimeError("Calibration contains Family=-1; calibration leakage detected.")

    agent = load_model()
    q = get_q_values(agent, X)

    q0 = q[:, 0]
    q1 = q[:, 1]

    predicted = np.argmax(q, axis=1)
    errors = predicted != y

    # 1. Existing absolute Q-margin.
    margin = np.abs(q1 - q0)
    anomaly_margin = -margin

    # 2. Maximum Q-value.
    max_q = np.max(q, axis=1)
    anomaly_max_q = -max_q

    # 3. Softmax entropy of Q-values.
    q_shifted = q - np.max(q, axis=1, keepdims=True)
    exp_q = np.exp(q_shifted)
    probabilities = exp_q / np.sum(exp_q, axis=1, keepdims=True)
    entropy = -np.sum(
        probabilities * np.log(probabilities + 1e-12),
        axis=1,
    )

    # 4. Energy-style score.
    # Higher = lower confidence / more uncertain.
    energy = -np.log(np.exp(q_shifted).sum(axis=1)) - np.max(q, axis=1)

    # 5. Scale-normalized Q-margin.
    normalized_margin = margin / (
        np.abs(q0) + np.abs(q1) + 1e-8
    )
    anomaly_normalized_margin = -normalized_margin

    scores = {
        "q_margin": {
            "score": anomaly_margin,
            "interpretation": "higher = more uncertain",
        },
        "negative_max_q": {
            "score": anomaly_max_q,
            "interpretation": "higher = more uncertain",
        },
        "q_entropy": {
            "score": entropy,
            "interpretation": "higher = more uncertain",
        },
        "q_energy": {
            "score": energy,
            "interpretation": "higher = more uncertain",
        },
        "negative_normalized_margin": {
            "score": anomaly_normalized_margin,
            "interpretation": "higher = more uncertain",
        },
    }

    results = {
        "experiment": "Phase 7 calibration-only open-set score comparison",
        "seed": SEED,
        "calibration_samples": int(len(df)),
        "calibration_benign": int((y == 0).sum()),
        "calibration_ransomware": int((y == 1).sum()),
        "calibration_errors": int(errors.sum()),
        "calibration_accuracy": float((~errors).mean()),
        "fixed_rejection_rate": REJECTION_RATE,
        "scores": {},
    }

    for name, info in scores.items():
        score = info["score"]

        if not np.isfinite(score).all():
            raise RuntimeError(f"Non-finite values detected for score: {name}")

        auc = roc_auc_score(errors.astype(int), score)

        threshold = percentile_threshold_for_rejection(
            score,
            REJECTION_RATE,
        )

        evaluation = evaluate_score(
            score,
            errors,
            threshold,
        )

        results["scores"][name] = {
            "interpretation": info["interpretation"],
            "roc_auc_for_error_detection": float(auc),
            **evaluation,
            "score_statistics": {
                "min": float(np.min(score)),
                "p05": float(np.percentile(score, 5)),
                "median": float(np.median(score)),
                "p95": float(np.percentile(score, 95)),
                "max": float(np.max(score)),
            },
        }

    # Select only using calibration-set evidence.
    ranking = sorted(
        results["scores"].items(),
        key=lambda item: (
            item[1]["incorrect_capture_rate"],
            item[1]["roc_auc_for_error_detection"],
        ),
        reverse=True,
    )

    results["selection"] = {
        "criterion": (
            "highest incorrect-prediction capture rate at exactly "
            "5% calibration rejection; ROC-AUC used as tie-break"
        ),
        "selected_score": ranking[0][0],
        "ranking": [name for name, _ in ranking],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 70)
    print("PHASE 7 — CALIBRATION-ONLY OPEN-SET SCORE COMPARISON")
    print("=" * 70)
    print(f"Calibration samples : {len(df)}")
    print(f"Calibration errors  : {errors.sum()}")
    print(f"Calibration accuracy: {results['calibration_accuracy']:.6f}")
    print()

    print(
        f"{'Score':30s}"
        f"{'ROC-AUC':>10s}"
        f"{'Reject %':>12s}"
        f"{'Error capture':>16s}"
        f"{'Remaining acc':>16s}"
    )
    print("-" * 84)

    for name, item in results["scores"].items():
        print(
            f"{name:30s}"
            f"{item['roc_auc_for_error_detection']:10.6f}"
            f"{item['rejection_rate'] * 100:11.3f}%"
            f"{item['incorrect_capture_rate'] * 100:15.3f}%"
            f"{item['remaining_accuracy'] * 100:15.3f}%"
        )

    print()
    print(f"SELECTED SCORE: {results['selection']['selected_score']}")
    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
