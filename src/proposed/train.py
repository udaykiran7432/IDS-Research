import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from agent import D3QNAgent
from data import load_data
from environment import IDSEnvironment
from replay_buffer import PrioritizedReplayBuffer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "phase5"
REPORT_DIR = PROJECT_ROOT / "reports" / "phase5"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(
    episodes=10,
    steps_per_episode=20_000,
    replay_capacity=100_000,
    batch_size=64,
    seed=42,
    device="cpu",
    artifact_dir=None,
    report_dir=None,
    per_alpha=0.6,
    per_beta_start=0.4,
    per_beta_increment=0.001,
):
    set_seed(seed)

    artifact_dir = (
        Path(artifact_dir) if artifact_dir else ARTIFACT_DIR
    )
    report_dir = (
        Path(report_dir) if report_dir else REPORT_DIR
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    (
        X_train,
        y_train,
        _,
        _,
        _,
        _,
    ) = load_data()

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

    checkpoint_path = (artifact_dir / "d3qn_per.pt").resolve()

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
        },
        checkpoint_path,
    )

    report = {
        "experiment": "d3qn_per",
        "method": "Dueling Double Deep Q-Network + Prioritized Experience Replay",
        "seed": seed,
        "device": str(agent.device),
        "state_dimension": X_train.shape[1],
        "action_dimension": 2,
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

    report_path = report_dir / "d3qn_per_training.json"

    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    print("\nD3QN + PER training complete.")
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
    parser.add_argument("--artifact-dir", type=str, default=None)
    parser.add_argument("--report-dir", type=str, default=None)
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
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
        artifact_dir=args.artifact_dir,
        report_dir=args.report_dir,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_increment=args.per_beta_increment,
    )


if __name__ == "__main__":
    main()
