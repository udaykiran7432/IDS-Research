import random

import numpy as np
import torch
from torch import nn, optim

from dqn import DQN


class DQNAgent:
    """DQN agent implementing the baseline learning configuration."""

    def __init__(
        self,
        state_dim=8,
        action_dim=2,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        batch_size=64,
        seed=42,
        device=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.random = random.Random(seed)

        if device is None:
            device = torch.device("cpu")

        self.device = torch.device(device)

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate,
        )

        self.loss_fn = nn.MSELoss()

    def select_action(self, state, training=True):
        """Select an action using epsilon-greedy policy."""

        if training and self.random.random() < self.epsilon:
            return self.random.randrange(self.action_dim)

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    def optimize(self, experiences):
        """Perform one DQN optimization step."""

        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.as_tensor(
            np.asarray(states),
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.as_tensor(
            actions,
            dtype=torch.int64,
            device=self.device,
        ).unsqueeze(1)

        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.as_tensor(
            np.asarray(next_states),
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        )

        current_q = self.policy_net(states).gather(
            1, actions
        ).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def decay_epsilon_step(self):
        """Apply epsilon decay after every training step."""

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )

    def decay_epsilon(self):
        """Legacy episode-level decay method."""

        self.decay_epsilon_step()

    def update_target_network(self):
        """Copy policy-network weights to target network."""

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )


if __name__ == "__main__":
    print("=== DQN Agent Test ===")

    torch.manual_seed(42)
    np.random.seed(42)

    agent = DQNAgent(device="cpu")

    print("Device:", agent.device)
    print("Gamma:", agent.gamma)
    print("Learning rate: 0.001")
    print("Batch size:", agent.batch_size)
    print("Initial epsilon:", agent.epsilon)

    state = np.zeros(8, dtype=np.float32)

    action = agent.select_action(
        state,
        training=True,
    )

    print("Selected action:", action)

    assert action in (0, 1)

    experiences = []

    for i in range(64):
        current_state = np.random.randn(8).astype(np.float32)
        next_state = np.random.randn(8).astype(np.float32)
        action = i % 2
        reward = 1.0 if action == 1 else 0.1
        done = i % 10 == 0

        experiences.append(
            (
                current_state,
                action,
                reward,
                next_state,
                done,
            )
        )

    loss = agent.optimize(experiences)

    print("Optimization loss:", loss)

    assert np.isfinite(loss)

    old_epsilon = agent.epsilon
    agent.decay_epsilon_step()

    print("Epsilon after step decay:", agent.epsilon)

    assert agent.epsilon < old_epsilon

    agent.update_target_network()

    print("Target network update: PASSED")
    print("DQN agent test: PASSED")
