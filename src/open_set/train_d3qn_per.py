import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch



PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROPOSED_DIR = PROJECT_ROOT / "src" / "proposed"

if str(PROPOSED_DIR) not in sys.path:
    sys.path.insert(0, str(PROPOSED_DIR))

from agent import D3QNAgent
from environment import IDSEnvironment
from replay_buffer import PrioritizedReplayBuffer

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase7_open_set"
    / "train.csv"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "phase7"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "phase7"
)

TARGET_COLUMN = "Prediction"

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_phase7_training_data():
    import pandas as pd

    df = pd.read_csv(TRAIN_PATH)

    required = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing Phase 7 training columns: {sorted(missing)}"
        )

    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.int64)

    if not np.isfinite(X).all():
        raise ValueError("Phase 7 training features contain non-finite values.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Phase 7 labels must contain only 0 and 1.")

    # Critical family-held-out check:
    # Locky is encoded as -1 and must never occur in training.
    family_column = df["Family"].to_numpy()

    if np.any(family_column == -1):
        raise ValueError(
            "Leakage detected: unseen-family encoding (-1) "
            "appears in Phase 7 training data."
        )

    return X, y


def train(
    episodes=10,
    steps_per_episode=20_000,
    replay_capacity=100_000,
    batch_size=64,
    seed=42,
    device="cpu",
    per_alpha=0.6,
    per_beta_start=0.4,
    per_beta_increment=0.001,
):
    set_seed(seed)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_phase7_training_data()

    print("=== PHASE 7 D3QN + PER ===")
    print("Training dataset:", TRAIN_PATH)
    print("Training samples:", len(X_train))
    print("State dimension:", X_train.shape[1])
    print("Benign samples:", int((y_train == 0).sum()))
    print("Ransomware samples:", int((y_train == 1).sum()))
    print("Unseen-family training samples: 0")
    print("Seed:", seed)
    print("Device:", device)
    print()

    env = IDSEnvironment(
        states=X_train,
        labels=y_train,
    )

    agent = D3QNAgent(
        state_dim=X_train.shape[1],
        action_dim=2,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        batch_size=batch_size,
        seed=seed,
        device=device,
        use_per=True,
    )

    replay_buffer = PrioritizedReplayBuffer(
        capacity=replay_capacity,
        seed=seed,
        alpha=per_alpha,
        beta_start=per_beta_start,
        beta_increment=per_beta_increment,
    )

    history = []
    total_steps = 0

    for episode in range(1, episodes + 1):
        state = env.reset(start_index=0)

        episode_reward = 0.0
        episode_losses = []

        for step in range(steps_per_episode):
            action = agent.select_action(
                state,
                training=True,
            )

            next_state, reward, environment_done, info = env.step(
                action
            )

            done = (
                step == steps_per_episode - 1
                or environment_done
            )

            replay_buffer.add(
                state,
                action,
                reward,
                next_state,
                done,
            )

            episode_reward += reward
            total_steps += 1

            if len(replay_buffer) >= batch_size:
                experiences, indices, weights = replay_buffer.sample(
                    batch_size
                )

                loss, td_errors = agent.optimize(
                    experiences,
                    indices=indices,
                    weights=weights,
                )

                replay_buffer.update_priorities(
                    indices,
                    td_errors,
                )

                episode_losses.append(loss)

            agent.decay_epsilon_step()

            state = next_state

            if environment_done:
                break

        agent.update_target_network()

        mean_loss = (
            float(np.mean(episode_losses))
            if episode_losses
            else None
        )

        episode_result = {
            "episode": episode,
            "steps_completed": step + 1,
            "total_steps": total_steps,
            "episode_reward": episode_reward,
            "mean_loss": mean_loss,
            "epsilon": agent.epsilon,
            "replay_size": len(replay_buffer),
            "per_beta": replay_buffer.beta,
        }

        history.append(episode_result)

        print(
            f"Episode {episode}/{episodes} | "
            f"steps={step + 1} | "
            f"reward={episode_reward:.4f} | "
            f"loss={mean_loss} | "
            f"epsilon={agent.epsilon:.6f} | "
            f"replay={len(replay_buffer)} | "
            f"beta={replay_buffer.beta:.6f}"
        )

    checkpoint_path = (
        ARTIFACT_DIR
        / f"d3qn_per_open_set_seed_{seed}.pt"
    )

    torch.save(
        {
            "model_state_dict": agent.policy_net.state_dict(),
            "target_model_state_dict": agent.target_net.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "epsilon": agent.epsilon,
            "episodes": episodes,
            "steps_per_episode": steps_per_episode,
            "seed": seed,
            "per_alpha": per_alpha,
            "per_beta_start": per_beta_start,
            "per_beta_increment": per_beta_increment,
            "feature_columns": FEATURE_COLUMNS,
            "training_dataset": str(
                TRAIN_PATH.relative_to(PROJECT_ROOT)
            ),
            "unseen_family": "Locky",
            "unseen_family_training_samples": 0,
        },
        checkpoint_path,
    )

    report = {
        "experiment": "phase7_d3qn_per_family_held_out",
        "method": (
            "Dueling Double Deep Q-Network "
            "+ Prioritized Experience Replay"
        ),
        "seed": seed,
        "device": str(agent.device),
        "training_dataset": str(
            TRAIN_PATH.relative_to(PROJECT_ROOT)
        ),
        "state_dimension": int(X_train.shape[1]),
        "action_dimension": 2,
        "training_samples": int(len(X_train)),
        "benign_samples": int((y_train == 0).sum()),
        "ransomware_samples": int((y_train == 1).sum()),
        "unseen_family": "Locky",
        "unseen_family_training_samples": 0,
        "episodes": episodes,
        "steps_per_episode": steps_per_episode,
        "total_steps": total_steps,
        "replay_capacity": replay_capacity,
        "batch_size": batch_size,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay": 0.995,
        "epsilon_decay_frequency": "every_step",
        "target_update": "end_of_episode",
        "loss": "importance_sampling_weighted_MSE",
        "optimizer": "Adam",
        "state_transition": "sequential_training_data",
        "architecture": {
            "type": "dueling_dqn",
            "feature_layers": [64, 64],
            "value_stream": [64, 1],
            "advantage_stream": [64, 2],
        },
        "target_rule": "double_dqn",
        "prioritized_replay": {
            "enabled": True,
            "alpha": per_alpha,
            "beta_start": per_beta_start,
            "beta_increment": per_beta_increment,
            "priority_epsilon": 1e-6,
            "priority_source": "absolute_td_error",
            "new_transition_priority": "maximum_current_priority",
        },
        "history": history,
        "checkpoint": str(
            checkpoint_path.relative_to(PROJECT_ROOT)
        ),
    }

    report_path = (
        REPORT_DIR
        / f"d3qn_per_open_set_seed_{seed}.json"
    )

    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    print()
    print("Phase 7 D3QN + PER training complete.")
    print("Checkpoint:", checkpoint_path)
    print("Report:", report_path)

    return agent, history


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=10)

    parser.add_argument(
        "--steps-per-episode",
        type=int,
        default=20_000,
    )

    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    parser.add_argument(
        "--per-alpha",
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "--per-beta-start",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--per-beta-increment",
        type=float,
        default=0.001,
    )

    args = parser.parse_args()

    train(
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_increment=args.per_beta_increment,
    )


if __name__ == "__main__":
    main()
