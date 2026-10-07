import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")

from evaluation.controlled_agents import ControlledQAgent


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase7_open_set"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "phase10"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "phase10"

SEED = 42
STATE_DIM = 8
ACTION_DIM = 2
DEVICE = "cpu"

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
    calibration = pd.read_csv(
        DATA_DIR / "calibration.csv"
    )

    test = pd.read_csv(
        DATA_DIR / "test.csv"
    )

    metadata = pd.read_csv(
        DATA_DIR / "test_metadata.csv"
    )

    X_cal = calibration[FEATURE_COLUMNS].values.astype(
        np.float32
    )
    y_cal = calibration["Prediction"].values.astype(
        np.int64
    )

    X_test = test[FEATURE_COLUMNS].values.astype(
        np.float32
    )
    y_test = test["Prediction"].values.astype(
        np.int64
    )

    return (
        X_cal,
        y_cal,
        X_test,
        y_test,
        metadata,
    )


def load_agent(config):
    checkpoint_path = (
        ARTIFACT_DIR
        / f"{config.lower()}_seed_{SEED}.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    agent = ControlledQAgent(
        config=config,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        seed=SEED,
        device=DEVICE,
    )

    agent.policy_net.load_state_dict(
        checkpoint["model_state_dict"]
    )

    agent.target_net.load_state_dict(
        checkpoint["target_state_dict"]
    )

    agent.epsilon = checkpoint["epsilon"]

    agent.policy_net.eval()
    agent.target_net.eval()

    return agent


def predict(agent, X):
    predictions = []

    with torch.no_grad():
        states = torch.as_tensor(
            X,
            dtype=torch.float32,
            device=agent.device,
        )

        q_values = agent.policy_net(states)

        predictions = (
            q_values.argmax(dim=1)
            .cpu()
            .numpy()
            .astype(np.int64)
        )

    return predictions


def classification_metrics(y_true, y_pred):
    tn = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 0)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 0)
        )
    )

    tp = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 1)
        )
    )

    total = tn + fp + fn + tp

    accuracy = (
        (tp + tn) / total
        if total
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn)
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "specificity": specificity,
        "confusion_matrix": [
            [tn, fp],
            [fn, tp],
        ],
    }


def evaluate_one(
    config,
    X_cal,
    y_cal,
    X_test,
    y_test,
    metadata,
):
    print("=" * 70)
    print(f"EVALUATING {config}")
    print("=" * 70)

    agent = load_agent(config)

    cal_predictions = predict(
        agent,
        X_cal,
    )

    test_predictions = predict(
        agent,
        X_test,
    )

    known_mask = (
        metadata["Family"]
        != "Locky"
    )

    locky_mask = (
        metadata["Family"]
        == "Locky"
    )

    known_mask = known_mask.to_numpy()
    locky_mask = locky_mask.to_numpy()

    known_metrics = classification_metrics(
        y_test[known_mask],
        test_predictions[known_mask],
    )

    all_metrics = classification_metrics(
        y_test,
        test_predictions,
    )

    locky_predictions = test_predictions[
        locky_mask
    ]

    locky_unknown = 0

    # E1/E2/E3 are closed-set classifiers.
    # They do not have an open-set rejection mechanism.
    #
    # Therefore every Locky sample is necessarily
    # assigned to one of the known classes.
    locky_known = int(
        len(locky_predictions)
    )

    result = {
        "experiment": config,
        "seed": SEED,
        "calibration_rows": int(len(X_cal)),
        "test_rows": int(len(X_test)),
        "known_test_rows": int(np.sum(known_mask)),
        "locky_test_rows": int(np.sum(locky_mask)),
        "calibration_prediction_counts": {
            "benign": int(
                np.sum(cal_predictions == 0)
            ),
            "ransomware": int(
                np.sum(cal_predictions == 1)
            ),
        },
        "all_test_classification": all_metrics,
        "known_family_classification": known_metrics,
        "locky_open_set": {
            "unknown": locky_unknown,
            "known": locky_known,
            "unknown_detection_rate": 0.0,
            "false_accept_rate": 1.0,
        },
    }

    print(
        f"Known accuracy:     "
        f"{known_metrics['accuracy']:.6f}"
    )

    print(
        f"Known precision:    "
        f"{known_metrics['precision']:.6f}"
    )

    print(
        f"Known recall:       "
        f"{known_metrics['recall']:.6f}"
    )

    print(
        f"Known F1:            "
        f"{known_metrics['f1']:.6f}"
    )

    print(
        f"Known FPR:           "
        f"{known_metrics['fpr']:.6f}"
    )

    print(
        f"Locky UNKNOWN:       "
        f"{locky_unknown}"
    )

    print(
        f"Locky accepted known:"
        f" {locky_known}"
    )

    report_path = (
        REPORT_DIR
        / f"{config.lower()}_evaluation_seed_{SEED}.json"
    )

    with open(report_path, "w") as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print(
        f"Saved report: {report_path}"
    )

    return result


def main():
    global SEED

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    SEED = args.seed

    print(
        "Loading Phase 7 family-held-out "
        "calibration/test data..."
    )

    (
        X_cal,
        y_cal,
        X_test,
        y_test,
        metadata,
    ) = load_data()

    print(
        f"Calibration rows: {len(X_cal)}"
    )

    print(
        f"Test rows:        {len(X_test)}"
    )

    print(
        f"Features:         {FEATURE_COLUMNS}"
    )

    print(
        f"Test shape:       {X_test.shape}"
    )

    assert len(X_test) == 62257
    assert len(X_cal) == 17358

    locky_count = int(
        np.sum(
            metadata["Family"].to_numpy()
            == "Locky"
        )
    )

    assert locky_count == 25062

    print(
        f"Locky test rows:  {locky_count}"
    )

    results = []

    for config in ("E1", "E2", "E3"):
        result = evaluate_one(
            config,
            X_cal,
            y_cal,
            X_test,
            y_test,
            metadata,
        )

        results.append(result)

    combined_path = (
        REPORT_DIR
        / f"controlled_evaluation_seed_{SEED}.json"
    )

    with open(combined_path, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("CONTROLLED EVALUATION COMPLETE")
    print("=" * 70)
    print(
        f"Combined report: {combined_path}"
    )


if __name__ == "__main__":
    main()