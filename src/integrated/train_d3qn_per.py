from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT / "src" / "proposed"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "open_set"))

from agent import D3QNAgent
from environment import IDSEnvironment
from replay_buffer import PrioritizedReplayBuffer


SEED = 42
STATE_DIM = 5
ACTION_DIM = 2

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase8_integrated"
    / "train.csv"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "phase8"
    / "d3qn_per_vae_seed_42.pt"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "phase8"
    / "d3qn_per_vae_training_seed_42.json"
)

FEATURE_COLUMNS = [
    "latent_0",
    "latent_1",
    "latent_2",
    "latent_3",
    "reconstruction_error",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    set_seed(SEED)

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(TRAIN_PATH)

    required = FEATURE_COLUMNS + ["Prediction"]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    if len(df) != 69428:
        raise RuntimeError(
            f"Unexpected training size: {len(df)}"
        )

    X = df[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float32)

    y = df[
        "Prediction"
    ].to_numpy(dtype=np.int64)

    if not np.isfinite(X).all():
        raise RuntimeError(
            "Training representation contains "
            "non-finite values."
        )

    if not np.isin(y, [0, 1]).all():
        raise RuntimeError(
            "Training target must be binary."
        )

    if X.shape[1] != STATE_DIM:
        raise RuntimeError(
            f"Expected state dimension {STATE_DIM}, "
            f"got {X.shape[1]}"
        )

    # Verify the authoritative Phase 7 training split.
    source_train = pd.read_csv(
        PROJECT_ROOT
        / "data"
        / "processed"
        / "phase7_open_set"
        / "train.csv"
    )

    unseen_family_count = int(
        (source_train["Family"] == -1).sum()
    )

    if unseen_family_count != 0:
        raise RuntimeError(
            "Unseen-family samples leaked into training: "
            f"{unseen_family_count}"
        )

    print("=== Phase 8 D3QN + PER Training ===")
    print("Seed:", SEED)
    print("Training samples:", len(df))
    print("State dimension:", STATE_DIM)
    print("Action dimension:", ACTION_DIM)
    print(
        "Unseen-family samples in training:",
        unseen_family_count,
    )
    print("Device: CPU")

    agent = D3QNAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        batch_size=64,
        seed=SEED,
        device=torch.device("cpu"),
        use_per=True,
    )

    replay_buffer = PrioritizedReplayBuffer(
        capacity=100000,
        alpha=0.6,
        beta_start=0.4,
        beta_increment=0.001,
        seed=SEED,
    )

    environment = IDSEnvironment(
        X,
        y,
    )

    num_episodes = 10
    max_steps_per_episode = 20000

    episode_results = []

    for episode in range(
        1,
        num_episodes + 1,
    ):
        state = environment.reset()

        total_reward = 0.0
        losses = []
        steps = 0

        for _ in range(
            max_steps_per_episode
        ):
            action = agent.select_action(
                state,
                training=True,
            )

            next_state, reward, done, info = (
                environment.step(action)
            )

            replay_buffer.add(
                state,
                action,
                reward,
                next_state,
                done,
            )

            if (
                len(replay_buffer)
                >= agent.batch_size
            ):
                (
                    experiences,
                    indices,
                    weights,
                ) = replay_buffer.sample(
                    agent.batch_size
                )

                loss, td_errors = agent.optimize(
                    experiences,
                    indices=indices,
                    weights=weights,
                )

                replay_buffer.update_priorities(
                    indices,
                    np.abs(td_errors),
                )

                losses.append(
                    float(loss)
                )

            agent.decay_epsilon_step()

            total_reward += float(reward)
            steps += 1
            state = next_state

            if done:
                break

        # Match the established training protocol:
        # synchronize the target network after each episode.
        agent.update_target_network()

        mean_loss = (
            float(np.mean(losses))
            if losses
            else None
        )

        result = {
            "episode": episode,
            "steps": steps,
            "total_reward": total_reward,
            "mean_loss": mean_loss,
            "epsilon": float(agent.epsilon),
            "replay_size": int(
                len(replay_buffer)
            ),
            "per_beta": float(
                replay_buffer.beta
            ),
        }

        episode_results.append(result)

        if mean_loss is None:
            print(
                f"Episode {episode}: "
                f"reward={total_reward:.1f}, "
                "loss=None"
            )
        else:
            print(
                f"Episode {episode}: "
                f"reward={total_reward:.1f}, "
                f"loss={mean_loss:.6f}"
            )

    checkpoint = {
        "experiment": (
            "Phase 8 D3QN + PER + VAE representation"
        ),
        "seed": SEED,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "model_state_dict": (
            agent.policy_net.state_dict()
        ),
        "target_state_dict": (
            agent.target_net.state_dict()
        ),
        "epsilon": float(agent.epsilon),
        "feature_columns": FEATURE_COLUMNS,
        "hyperparameters": {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "epsilon_decay": 0.995,
            "batch_size": 64,
            "per_alpha": 0.6,
            "per_beta_start": 0.4,
            "per_beta_increment": 0.001,
            "per_capacity": 100000,
        },
    }

    torch.save(
        checkpoint,
        CHECKPOINT_PATH,
    )

    report = {
        "experiment": (
            "Phase 8 D3QN + PER + VAE representation"
        ),
        "seed": SEED,
        "training_samples": len(df),
        "state_dimension": STATE_DIM,
        "feature_columns": FEATURE_COLUMNS,
        "unseen_family_training_samples": (
            unseen_family_count
        ),
        "calibration_data_used": False,
        "test_data_used": False,
        "episodes": num_episodes,
        "max_steps_per_episode": (
            max_steps_per_episode
        ),
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "episodes_results": episode_results,
    }

    with REPORT_PATH.open("w") as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print("=== Training Complete ===")
    print(
        "Final epsilon:",
        agent.epsilon,
    )
    print(
        "Replay size:",
        len(replay_buffer),
    )
    print(
        "PER beta:",
        replay_buffer.beta,
    )
    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )
    print(
        "Report:",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()
