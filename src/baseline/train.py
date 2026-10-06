import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from agent import DQNAgent
from data import load_data
from environment import IDSEnvironment
from replay_buffer import ReplayBuffer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "phase2"
REPORT_DIR = PROJECT_ROOT / "reports" / "phase2"


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
):
    set_seed(seed)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

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

    agent = DQNAgent(
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
    )

    replay_buffer = ReplayBuffer(
        capacity=replay_capacity,
        seed=seed,
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

            next_state, reward, environment_done, info = env.step(action)

            # The paper defines an episode by its configured
            # number of steps. Therefore the final step of the
            # episode is treated as terminal for the TD target.
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
                experiences = replay_buffer.sample(batch_size)
                loss = agent.optimize(experiences)
                episode_losses.append(loss)

            # Paper: epsilon decays after every step.
            agent.decay_epsilon_step()

            state = next_state

            if environment_done:
                break

        # Paper: target network synchronization at episode end.
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
        }

        history.append(episode_result)

        print(
            f"Episode {episode}/{episodes} | "
            f"steps={step + 1} | "
            f"reward={episode_reward:.4f} | "
            f"loss={mean_loss} | "
            f"epsilon={agent.epsilon:.6f} | "
            f"replay={len(replay_buffer)}"
        )

    checkpoint_path = ARTIFACT_DIR / "baseline_dqn.pt"

    torch.save(
        {
            "model_state_dict": agent.policy_net.state_dict(),
            "target_model_state_dict": agent.target_net.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "epsilon": agent.epsilon,
            "episodes": episodes,
            "steps_per_episode": steps_per_episode,
            "seed": seed,
        },
        checkpoint_path,
    )

    report = {
        "experiment": "baseline_dqn",
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
        "loss": "MSE",
        "optimizer": "Adam",
        "state_transition": "sequential_training_data",
        "history": history,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
    }

    report_path = REPORT_DIR / "baseline_training.json"

    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    print("\nTraining complete.")
    print("Checkpoint:", checkpoint_path)
    print("Report:", report_path)

    return agent, history


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps-per-episode", type=int, default=20_000)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    train(
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
