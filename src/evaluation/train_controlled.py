import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")

from evaluation.controlled_agents import ControlledQAgent
from proposed.environment import IDSEnvironment


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "phase7_open_set" / "train.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "phase10"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "phase10"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


SEED = 42
STATE_DIM = 8
ACTION_DIM = 2

EPISODES = 10
STEPS_PER_EPISODE = 20_000

DEVICE = "cpu"


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_training_data():
    df = pd.read_csv(TRAIN_PATH)

    feature_columns = [
        "Time",
        "Protcol",
        "Flag",
        "Family",
        "Clusters",
        "Threats",
        "USD",
        "BTC",
    ]

    X = df[feature_columns].values.astype(np.float32)
    y = df["Prediction"].values.astype(np.int64)

    return X, y, feature_columns


def train_experiment(config: str, X: np.ndarray, y: np.ndarray):
    print("=" * 70)
    print(f"STARTING {config}")
    print("=" * 70)

    set_seed(SEED)

    environment = IDSEnvironment(
    states=X,
    labels=y,
)

    agent = ControlledQAgent(
        config=config,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        seed=SEED,
        device=DEVICE,
    )

    episode_results = []

    for episode in range(EPISODES):
        state = environment.reset()

        total_reward = 0.0
        losses = []
        steps = 0

        done = False

        while not done and steps < STEPS_PER_EPISODE:
            action = agent.select_action(
                state,
                training=True,
            )

            next_state, reward, done, info = environment.step(action)

            if config == "E3":
                agent.replay_buffer.add(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
    )

                if len(agent.replay_buffer) >= agent.batch_size:
                    experiences, indices, weights = (
                        agent.replay_buffer.sample(
                            agent.batch_size
            )
        )

                    loss, td_errors = agent.optimize_per(
                        experiences,
                        weights,
        )

                    agent.replay_buffer.update_priorities(
                        indices,
                        td_errors,
        )

                    losses.append(loss)

            else:
                agent.replay_buffer.add(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )

                if len(agent.replay_buffer) >= agent.batch_size:
                    experiences = agent.replay_buffer.sample(
                        agent.batch_size
                    )

                    loss = agent.optimize(experiences)

                    if loss is not None:
                        losses.append(loss)

            state = next_state
            total_reward += reward
            steps += 1

            agent.decay_epsilon_step()

            if done:
                break

        agent.update_target_network()

        mean_loss = (
            float(np.mean(losses))
            if losses
            else None
        )

        episode_result = {
            "episode": episode + 1,
            "steps": steps,
            "total_reward": float(total_reward),
            "mean_loss": mean_loss,
            "epsilon": float(agent.epsilon),
        }

        episode_results.append(episode_result)

        print(
            f"{config} | "
            f"episode={episode + 1}/{EPISODES} | "
            f"steps={steps} | "
            f"reward={total_reward:.4f} | "
            f"loss={mean_loss} | "
            f"epsilon={agent.epsilon:.6f}"
        )

    checkpoint_path = (
        ARTIFACT_DIR /
        f"{config.lower()}_seed_{SEED}.pt"
    )

    torch.save(
        {
            "experiment": config,
            "seed": SEED,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "model_state_dict": agent.policy_net.state_dict(),
            "target_state_dict": agent.target_net.state_dict(),
            "epsilon": agent.epsilon,
            "feature_columns": [
                "Time",
                "Protcol",
                "Flag",
                "Family",
                "Clusters",
                "Threats",
                "USD",
                "BTC",
            ],
            "hyperparameters": {
                "episodes": EPISODES,
                "steps_per_episode": STEPS_PER_EPISODE,
                "batch_size": agent.batch_size,
                "gamma": agent.gamma,
                "learning_rate": agent.optimizer.param_groups[0]["lr"],
                "seed": SEED,
                "device": DEVICE,
            },
        },
        checkpoint_path,
    )

    result = {
        "experiment": config,
        "seed": SEED,
        "train_rows": int(len(X)),
        "state_dim": STATE_DIM,
        "episodes": EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "final_epsilon": float(agent.epsilon),
        "episode_results": episode_results,
        "checkpoint": str(checkpoint_path),
    }

    report_path = (
        REPORT_DIR /
        f"{config.lower()}_training_seed_{SEED}.json"
    )

    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved checkpoint: {checkpoint_path}")
    print(f"Saved report:     {report_path}")

    return result


def main():
    global SEED

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    SEED = args.seed

    print("\nLoading Phase 7 family-held-out training data...")
    X, y, feature_columns = load_training_data()

    print(f"Training rows: {len(X)}")
    print(f"Features: {feature_columns}")
    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Critical leakage check:
    # Family=-1 represents unseen-family encoding and must not occur
    # in the family-held-out training data.
    family_index = feature_columns.index("Family")

    family_unknown_count = int(
        np.sum(X[:, family_index] == -1)
    )

    print(
        f"Family=-1 training rows: "
        f"{family_unknown_count}"
    )

    assert family_unknown_count == 0, (
        "LEAKAGE CHECK FAILED: "
        "Family=-1 found in training data."
    )

    all_results = []

    for config in ("E1", "E2", "E3"):
        result = train_experiment(
            config,
            X,
            y,
        )

        all_results.append(result)

    combined_path = (
        REPORT_DIR /
        f"controlled_training_seed_{SEED}.json"
    )

    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("CONTROLLED TRAINING COMPLETE")
    print("=" * 70)
    print(f"Combined report: {combined_path}")


if __name__ == "__main__":
    main()