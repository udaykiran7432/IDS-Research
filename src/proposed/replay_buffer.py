from collections import deque
import random


class ReplayBuffer:
    """Fixed-size uniform experience replay buffer."""

    def __init__(self, capacity: int, seed: int = 42):
        if capacity <= 0:
            raise ValueError(
                "capacity must be positive"
            )

        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.random = random.Random(seed)

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        self.buffer.append(
            (
                state,
                action,
                reward,
                next_state,
                done,
            )
        )

    def sample(self, batch_size: int):
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} experiences "
                f"from buffer containing "
                f"{len(self.buffer)} experiences."
            )

        return self.random.sample(
            self.buffer,
            batch_size,
        )

    def __len__(self):
        return len(self.buffer)
