import torch
from torch import nn


class DQN(nn.Module):
    """Baseline Deep Q-Network."""

    def __init__(self, state_dim: int = 8, action_dim: int = 2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state):
        return self.network(state)


if __name__ == "__main__":
    torch.manual_seed(42)

    model = DQN()

    print("=== DQN Network Test ===")
    print(model)

    x = torch.randn(4, 8)
    q_values = model(x)

    print("\nInput shape :", x.shape)
    print("Output shape:", q_values.shape)
    print("\nQ-values:")
    print(q_values)

    assert q_values.shape == (4, 2)

    print("\nDQN network test: PASSED")
