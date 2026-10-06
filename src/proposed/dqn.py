import torch
from torch import nn


class D3QN(nn.Module):
    """Dueling Deep Q-Network used by the proposed D3QN method."""

    def __init__(self, state_dim: int = 8, action_dim: int = 2):
        super().__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state):
        features = self.feature_layer(state)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Dueling aggregation:
        # Q(s,a) = V(s) + [A(s,a) - mean(A(s,a))]
        q_values = value + (
            advantage - advantage.mean(dim=1, keepdim=True)
        )

        return q_values


if __name__ == "__main__":
    torch.manual_seed(42)

    model = D3QN()

    print("=== D3QN Network Test ===")
    print(model)

    x = torch.randn(4, 8)
    q_values = model(x)

    print("\nInput shape :", x.shape)
    print("Output shape:", q_values.shape)

    assert q_values.shape == (4, 2)

    # Verify the dueling aggregation numerically.
    with torch.no_grad():
        features = model.feature_layer(x)
        value = model.value_stream(features)
        advantage = model.advantage_stream(features)

        expected_q = value + (
            advantage - advantage.mean(dim=1, keepdim=True)
        )

    assert torch.allclose(q_values, expected_q)

    print("Dueling aggregation test: PASSED")
    print("D3QN network test: PASSED")
