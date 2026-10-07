import sys
from pathlib import Path

import json
import random
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "evaluation"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "proposed"))

from src.vae.model import VAE
from controlled_agents import ControlledQAgent
from proposed.environment import IDSEnvironment


SEED = 42

VAE_CHECKPOINT = (
    PROJECT_ROOT / "artifacts/phase10/vae_no_family_seed_42.pt"
)

TRAIN_PATH = (
    PROJECT_ROOT / "data/processed/phase7_open_set/train.csv"
)

OUTPUT_CHECKPOINT = (
    PROJECT_ROOT / "artifacts/phase10/e4b_d3qn_per_seed_42.pt"
)

REPORT_PATH = (
    PROJECT_ROOT / "reports/phase10/e4b_d3qn_per_training_seed_42.json"
)

FEATURE_COLUMNS = [
    "Time",
    "Protcol",
    "Flag",
    "Clusters",
    "Threats",
    "USD",
    "BTC",
]


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)
print("Seed:", SEED)


# ---------------------------------------------------------
# Load Phase-7 training data
# ---------------------------------------------------------
train_df = pd.read_csv(TRAIN_PATH)

X = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

# Phase-7 Prediction is already numerically encoded:
# 1 = benign (S)
# 0 = ransomware (A)
# 2 = ransomware (SS)
#
# Therefore:
# Prediction == 1 -> 0
# Prediction in {0,2} -> 1
raw_labels = train_df["Prediction"].to_numpy()

y = np.where(
    raw_labels == 1,
    0,
    1,
).astype(np.int64)

print()
print("Training data shape:", X.shape)
print("Raw Prediction distribution:")
print(
    pd.Series(raw_labels)
    .value_counts()
    .sort_index()
    .to_dict()
)

print("Binary label distribution:")
print(
    pd.Series(y)
    .value_counts()
    .sort_index()
    .to_dict()
)

assert X.shape[1] == 7
assert "Family" not in FEATURE_COLUMNS
assert set(np.unique(y)).issubset({0, 1})
assert len(np.unique(y)) == 2


# ---------------------------------------------------------
# Load E4b VAE
# ---------------------------------------------------------
vae_ckpt = torch.load(
    VAE_CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

assert vae_ckpt["input_dim"] == 7
assert "Family" not in vae_ckpt["feature_columns"]

vae = VAE(
    input_dim=vae_ckpt["input_dim"],
    hidden_dim=vae_ckpt["hidden_dim"],
    latent_dim=vae_ckpt["latent_dim"],
)

vae.load_state_dict(vae_ckpt["model_state_dict"])
vae.to(DEVICE)
vae.eval()

print()
print("E4b VAE loaded.")
print("Input dim:", vae_ckpt["input_dim"])
print("Latent dim:", vae_ckpt["latent_dim"])


# ---------------------------------------------------------
# Create deterministic representation
# latent mean + reconstruction error
# ---------------------------------------------------------
with torch.no_grad():
    tensor_x = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE,
    )

    mu, _ = vae.encode(tensor_x)
    reconstruction = vae.decode(mu)

    reconstruction_error = torch.mean(
        (reconstruction - tensor_x) ** 2,
        dim=1,
        keepdim=True,
    )

    representations = torch.cat(
        [mu, reconstruction_error],
        dim=1,
    ).cpu().numpy().astype(np.float32)


print()
print("Representation shape:", representations.shape)
print("Expected shape:", (len(X), 5))

assert representations.shape == (len(X), 5)


# ---------------------------------------------------------
# Existing validated Phase-10 E3 implementation
# ---------------------------------------------------------
agent = ControlledQAgent(
    config="E3",
    state_dim=5,
    action_dim=2,
    learning_rate=0.001,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=0.995,
    batch_size=64,
    seed=SEED,
    device=DEVICE,
    replay_capacity=100_000,
    per_alpha=0.6,
    per_beta_start=0.4,
    per_beta_increment=0.001,
)

env = IDSEnvironment(
    representations,
    y,
)


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------
NUM_EPISODES = 10
STEPS_PER_EPISODE = 20000

episode_rewards = []
episode_losses = []
episode_epsilons = []

print()
print("=" * 70)
print("E4b D3QN + PER TRAINING")
print("=" * 70)


for episode in range(NUM_EPISODES):

    state = env.reset(start_index=0)

    total_reward = 0.0
    losses = []

    for step in range(STEPS_PER_EPISODE):

        action = agent.select_action(state)

        next_state, reward, done, info = env.step(action)

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

            losses.append(float(loss))

        state = next_state
        total_reward += reward

        agent.epsilon = max(
            agent.epsilon_end,
            agent.epsilon * agent.epsilon_decay,
        )

        if done:
            state = env.reset(start_index=0)

    # Match Phase-10 controlled training:
    # target network synchronization at episode boundary.
    agent.target_net.load_state_dict(
        agent.policy_net.state_dict()
    )

    mean_loss = (
        float(np.mean(losses))
        if losses
        else None
    )

    episode_rewards.append(float(total_reward))
    episode_losses.append(mean_loss)
    episode_epsilons.append(float(agent.epsilon))

    print(
        f"Episode {episode + 1:02d}/{NUM_EPISODES} | "
        f"reward={total_reward:.1f} | "
        f"loss={mean_loss} | "
        f"epsilon={agent.epsilon:.6f}"
    )


# ---------------------------------------------------------
# Save checkpoint
# ---------------------------------------------------------
checkpoint = {
    "experiment": "E4b",
    "seed": SEED,
    "state_dim": 5,
    "action_dim": 2,
    "feature_columns": FEATURE_COLUMNS,
    "representation_columns": [
        "latent_0",
        "latent_1",
        "latent_2",
        "latent_3",
        "reconstruction_error",
    ],
    "vae_checkpoint": str(
        VAE_CHECKPOINT.relative_to(PROJECT_ROOT)
    ),
    "model_state_dict": agent.policy_net.state_dict(),
    "target_state_dict": agent.target_net.state_dict(),
    "epsilon": float(agent.epsilon),
    "episode_rewards": episode_rewards,
    "episode_losses": episode_losses,
    "episode_epsilons": episode_epsilons,
    "hyperparameters": {
        "num_episodes": NUM_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay": 0.995,
        "batch_size": 64,
        "use_per": True,
        "per_alpha": 0.6,
        "per_beta_start": 0.4,
        "per_beta_increment": 0.001,
        "replay_capacity": 100_000,
    },
}

OUTPUT_CHECKPOINT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

torch.save(
    checkpoint,
    OUTPUT_CHECKPOINT,
)


# ---------------------------------------------------------
# Save report
# ---------------------------------------------------------
report = {
    "experiment": "E4b",
    "seed": SEED,
    "device": str(DEVICE),
    "input_features": FEATURE_COLUMNS,
    "input_dim": 7,
    "representation_dim": 5,
    "representation": [
        "latent_0",
        "latent_1",
        "latent_2",
        "latent_3",
        "reconstruction_error",
    ],
    "train_size": int(len(X)),
    "raw_prediction_distribution": {
        str(k): int(v)
        for k, v in (
            pd.Series(raw_labels)
            .value_counts()
            .sort_index()
            .items()
        )
    },
    "binary_label_distribution": {
        str(k): int(v)
        for k, v in (
            pd.Series(y)
            .value_counts()
            .sort_index()
            .items()
        )
    },
    "episode_rewards": episode_rewards,
    "episode_losses": episode_losses,
    "episode_epsilons": episode_epsilons,
    "final_training": {
        "episode": NUM_EPISODES,
        "reward": episode_rewards[-1],
        "loss": episode_losses[-1],
        "epsilon": episode_epsilons[-1],
    },
    "checkpoint": str(
        OUTPUT_CHECKPOINT.relative_to(PROJECT_ROOT)
    ),
}

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)


print()
print("=" * 70)
print("E4b TRAINING COMPLETE")
print("=" * 70)
print("Checkpoint:", OUTPUT_CHECKPOINT)
print("Report:", REPORT_PATH)
