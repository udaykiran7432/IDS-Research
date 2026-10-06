import random

import numpy as np
import torch
from torch import nn, optim

from dqn import D3QN


class D3QNAgent:
    """D3QN agent: Dueling architecture + Double-DQN learning + optional PER."""

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
        use_per=False,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.use_per = use_per

        self.random = random.Random(seed)

        if device is None:
            device = torch.device("cpu")

        self.device = torch.device(device)

        self.policy_net = D3QN(
            state_dim,
            action_dim,
        ).to(self.device)

        self.target_net = D3QN(
            state_dim,
            action_dim,
        ).to(self.device)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate,
        )

        # Keep reduction='none' so PER can apply an IS weight to
        # each transition before averaging the minibatch loss.
        self.loss_fn = nn.MSELoss(reduction="none")

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

        return int(
            torch.argmax(q_values, dim=1).item()
        )

    def optimize(self, experiences, indices=None, weights=None):
        """
        Perform one Double-DQN optimization step.

        For PER:
            experiences = sampled transitions
            indices = replay-buffer indices
            weights = importance-sampling weights

        Returns:
            If PER is disabled: loss
            If PER is enabled: (loss, td_errors)
        """

        states, actions, rewards, next_states, dones = zip(
            *experiences
        )

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
            1,
            actions,
        ).squeeze(1)

        with torch.no_grad():
            # Double-DQN:
            # The policy network selects the next action.
            next_actions = self.policy_net(
                next_states
            ).argmax(
                dim=1,
                keepdim=True,
            )

            # The target network evaluates that action.
            next_q = self.target_net(
                next_states
            ).gather(
                1,
                next_actions,
            ).squeeze(1)

            target_q = (
                rewards
                + self.gamma * next_q * (1.0 - dones)
            )

        td_errors = target_q - current_q

        per_sample_loss = self.loss_fn(
            current_q,
            target_q,
        )

        if self.use_per:
            if weights is None:
                raise ValueError(
                    "PER optimization requires importance-sampling weights."
                )

            weights_tensor = torch.as_tensor(
                weights,
                dtype=torch.float32,
                device=self.device,
            )

            loss = (
                weights_tensor * per_sample_loss
            ).mean()
        else:
            loss = per_sample_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.use_per:
            return (
                float(loss.item()),
                td_errors.detach().cpu().numpy(),
            )

        return float(loss.item())

    def decay_epsilon_step(self):
        """Apply epsilon decay after every training step."""

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )

    def update_target_network(self):
        """Synchronize target network with policy network."""

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )


if __name__ == "__main__":
    print("=== D3QN + PER Agent Test ===")

    torch.manual_seed(42)
    np.random.seed(42)

    agent = D3QNAgent(
        device="cpu",
        use_per=True,
    )

    print("Device:", agent.device)
    print("Gamma:", agent.gamma)
    print("Learning rate: 0.001")
    print("Batch size:", agent.batch_size)
    print("Initial epsilon:", agent.epsilon)
    print("PER enabled:", agent.use_per)

    state = np.zeros(
        8,
        dtype=np.float32,
    )

    action = agent.select_action(
        state,
        training=True,
    )

    print("Selected action:", action)
    assert action in (0, 1)

    experiences = []

    for i in range(64):
        current_state = np.random.randn(
            8
        ).astype(np.float32)

        next_state = np.random.randn(
            8
        ).astype(np.float32)

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

    # Smoke-test the PER optimization interface.
    weights = np.ones(64, dtype=np.float32)

    loss, td_errors = agent.optimize(
        experiences,
        indices=list(range(64)),
        weights=weights,
    )

    print("Optimization loss:", loss)
    print("TD-error count:", len(td_errors))

    assert np.isfinite(loss)
    assert td_errors.shape == (64,)
    assert np.all(np.isfinite(td_errors))

    old_epsilon = agent.epsilon

    agent.decay_epsilon_step()

    print(
        "Epsilon after step decay:",
        agent.epsilon,
    )

    assert agent.epsilon < old_epsilon

    agent.update_target_network()

    print("Target network update: PASSED")
    print("D3QN + PER agent test: PASSED")
