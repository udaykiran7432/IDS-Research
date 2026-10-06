import random
from collections import deque
import numpy as np


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer.

    Sampling probability:
        p_i = priority_i ** alpha / sum_j(priority_j ** alpha)

    Importance-sampling weight:
        w_i = (N * p_i) ** (-beta)
        normalized by the maximum sampled weight.

    The buffer stores the same 5-tuple transition used by the existing
    D3QN implementation:
        (state, action, reward, next_state, done)
    """

    def __init__(
        self,
        capacity: int,
        seed: int = 42,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if not 0.0 <= beta_start <= 1.0:
            raise ValueError("beta_start must be in [0, 1]")
        if beta_increment < 0.0:
            raise ValueError("beta_increment must be non-negative")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.random = random.Random(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a transition with maximum current priority."""
        max_priority = max(self.priorities, default=1.0)

        self.buffer.append(
            (state, action, reward, next_state, done)
        )
        self.priorities.append(float(max_priority))

    def sample(self, batch_size: int):
        """
        Sample a prioritized minibatch.

        Returns:
            experiences: list of transitions
            indices: list of buffer indices
            weights: np.ndarray of normalized IS weights
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} experiences "
                f"from buffer containing {len(self.buffer)} experiences."
            )

        priorities = np.asarray(self.priorities, dtype=np.float64)
        scaled = np.maximum(priorities, self.epsilon) ** self.alpha

        total = scaled.sum()
        if not np.isfinite(total) or total <= 0.0:
            probabilities = np.full(
                len(self.buffer),
                1.0 / len(self.buffer),
                dtype=np.float64,
            )
        else:
            probabilities = scaled / total

        indices = self.random.choices(
            range(len(self.buffer)),
            weights=probabilities.tolist(),
            k=batch_size,
        )

        experiences = [self.buffer[index] for index in indices]

        # Importance-sampling correction.
        sample_probabilities = probabilities[indices]
        weights = (len(self.buffer) * sample_probabilities) ** (-self.beta)
        weights /= weights.max()

        # Anneal beta toward 1 over optimization steps.
        self.beta = min(1.0, self.beta + self.beta_increment)

        return experiences, indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        """Update sampled transition priorities from absolute TD errors."""
        td_errors = np.asarray(td_errors, dtype=np.float64).reshape(-1)

        if len(indices) != len(td_errors):
            raise ValueError(
                "indices and td_errors must have the same length."
            )

        for index, td_error in zip(indices, td_errors):
            if not 0 <= index < len(self.buffer):
                raise IndexError(f"Invalid replay-buffer index: {index}")

            priority = abs(float(td_error)) + self.epsilon
            self.priorities[index] = priority

    def __len__(self):
        return len(self.buffer)
