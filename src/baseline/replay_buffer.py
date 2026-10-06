from collections import deque
import random


class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity: int, seed: int = 42):
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.random = random.Random(seed)

    def add(self, state, action, reward, next_state, done):
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
                f"from buffer containing {len(self.buffer)} experiences."
            )

        return self.random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    buffer = ReplayBuffer(capacity=5, seed=42)

    for i in range(5):
        buffer.add(
            state=i,
            action=i % 2,
            reward=1.0,
            next_state=i + 1,
            done=False,
        )

    print("=== Replay Buffer Test ===")
    print("Capacity:", buffer.capacity)
    print("Length:", len(buffer))

    sample = buffer.sample(3)

    print("Sample size:", len(sample))
    print("Sample:", sample)

    buffer.add(
        state=99,
        action=1,
        reward=-1.0,
        next_state=100,
        done=True,
    )

    print("Length after overflow:", len(buffer))

    assert len(buffer) == 5
    assert len(sample) == 3

    print("Replay buffer test: PASSED")
