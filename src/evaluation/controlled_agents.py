from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn, optim

from baseline.dqn import DQN
from baseline.replay_buffer import ReplayBuffer
from proposed.dqn import D3QN
from proposed.replay_buffer import PrioritizedReplayBuffer


class ControlledQAgent:
    """
    Controlled RL agent used by Phase 10 E1-E3.

    E1:
        DQN + uniform replay + standard DQN target

    E2:
        D3QN + uniform replay + Double-DQN target

    E3:
        D3QN + PER + Double-DQN target

    The training hyperparameters are kept identical across
    configurations unless the configuration explicitly requires
    a replay-specific parameter.
    """

    VALID_CONFIGS = {"E1", "E2", "E3"}

    def __init__(
        self,
        config: str,
        state_dim: int,
        action_dim: int = 2,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        seed: int = 42,
        device: str | torch.device = "cpu",
        replay_capacity: int = 100_000,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_increment: float = 0.001,
    ):
        config = config.upper()

        if config not in self.VALID_CONFIGS:
            raise ValueError(
                f"Unknown configuration {config!r}. "
                f"Expected one of {sorted(self.VALID_CONFIGS)}."
            )

        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.seed = seed

        self.random = random.Random(seed)

        self.device = torch.device(device)

        # ---------------------------------------------------------
        # Network
        # ---------------------------------------------------------
        if config == "E1":
            self.policy_net = DQN(
                state_dim=state_dim,
                action_dim=action_dim,
            ).to(self.device)

            self.target_net = DQN(
                state_dim=state_dim,
                action_dim=action_dim,
            ).to(self.device)

        else:
            self.policy_net = D3QN(
                state_dim=state_dim,
                action_dim=action_dim,
            ).to(self.device)

            self.target_net = D3QN(
                state_dim=state_dim,
                action_dim=action_dim,
            ).to(self.device)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate,
        )

        self.loss_fn = nn.MSELoss(reduction="none")

        # ---------------------------------------------------------
        # Replay
        # ---------------------------------------------------------
        if config == "E3":
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=replay_capacity,
                seed=seed,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_increment=per_beta_increment,
            )
        else:
            self.replay_buffer = ReplayBuffer(
                capacity=replay_capacity,
                seed=seed,
            )

    @property
    def uses_double_dqn(self) -> bool:
        return self.config in {"E2", "E3"}

    @property
    def uses_per(self) -> bool:
        return self.config == "E3"

    def select_action(
        self,
        state,
        training: bool = True,
    ) -> int:
        """Select an epsilon-greedy action."""

        if (
            training
            and self.random.random() < self.epsilon
        ):
            return self.random.randrange(self.action_dim)

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(
                state_tensor
            )

        return int(
            torch.argmax(
                q_values,
                dim=1,
            ).item()
        )

    def optimize(self, experiences):
        """
        Perform one controlled optimization step.

        E1:
            target_net selects max next Q.

        E2/E3:
            policy_net selects next action,
            target_net evaluates that action.

        Returns:
            E1/E2: loss
            E3: (loss, td_errors)
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

        current_q = (
            self.policy_net(states)
            .gather(1, actions)
            .squeeze(1)
        )

        with torch.no_grad():
            if self.uses_double_dqn:
                next_actions = (
                    self.policy_net(next_states)
                    .argmax(
                        dim=1,
                        keepdim=True,
                    )
                )

                next_q = (
                    self.target_net(next_states)
                    .gather(
                        1,
                        next_actions,
                    )
                    .squeeze(1)
                )

            else:
                next_q = (
                    self.target_net(next_states)
                    .max(dim=1)
                    .values
                )

            target_q = (
                rewards
                + self.gamma * next_q
                * (1.0 - dones)
            )

        td_errors = target_q - current_q

        per_sample_loss = self.loss_fn(
            current_q,
            target_q,
        )

        if self.uses_per:
            # PER requires importance-sampling weights.
            experiences_are_per = isinstance(
                self.replay_buffer,
                PrioritizedReplayBuffer,
            )

            if not experiences_are_per:
                raise RuntimeError(
                    "E3 requires PrioritizedReplayBuffer."
                )

            raise RuntimeError(
                "Use optimize_per() for E3 so that "
                "PER indices and weights are explicit."
            )

        loss = per_sample_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def optimize_per(
        self,
        experiences,
        weights,
    ):
        """Perform a PER-weighted optimization step for E3."""

        if not self.uses_per:
            raise RuntimeError(
                "optimize_per() is only valid for E3."
            )

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

        current_q = (
            self.policy_net(states)
            .gather(1, actions)
            .squeeze(1)
        )

        with torch.no_grad():
            next_actions = (
                self.policy_net(next_states)
                .argmax(
                    dim=1,
                    keepdim=True,
                )
            )

            next_q = (
                self.target_net(next_states)
                .gather(
                    1,
                    next_actions,
                )
                .squeeze(1)
            )

            target_q = (
                rewards
                + self.gamma * next_q
                * (1.0 - dones)
            )

        td_errors = target_q - current_q

        per_sample_loss = self.loss_fn(
            current_q,
            target_q,
        )

        weights_tensor = torch.as_tensor(
            weights,
            dtype=torch.float32,
            device=self.device,
        )

        loss = (
            weights_tensor * per_sample_loss
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return (
            float(loss.item()),
            td_errors.detach().cpu().numpy(),
        )

    def decay_epsilon_step(self) -> None:
        """Apply epsilon decay after every training step."""

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )

    def update_target_network(self) -> None:
        """Synchronize target network with policy network."""

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )