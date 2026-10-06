import numpy as np


class IDSEnvironment:
    """
    Sequential dataset-backed IDS environment.

    Labels:
        0 = benign
        1 = ransomware

    Actions:
        0 = benign
        1 = ransomware

    Rewards:
        TP = +1.0
        TN = +0.1
        FP = -0.5
        FN = -1.0
    """

    TP_REWARD = 1.0
    TN_REWARD = 0.1
    FP_REWARD = -0.5
    FN_REWARD = -1.0

    def __init__(self, states, labels):
        states = np.asarray(states, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)

        if states.ndim != 2:
            raise ValueError("states must be a 2D array")

        if labels.ndim != 1:
            raise ValueError("labels must be a 1D array")

        if len(states) != len(labels):
            raise ValueError("states and labels must have the same length")

        if not np.isin(labels, [0, 1]).all():
            raise ValueError("labels must contain only 0 and 1")

        self.states = states
        self.labels = labels
        self.current_index = 0

    def reset(self, start_index=0):
        """Start an episode at a specified dataset index."""

        if not 0 <= start_index < len(self.states):
            raise ValueError("start_index is outside the dataset")

        self.current_index = start_index
        return self.states[self.current_index]

    def step(self, action):
        """Apply action and advance to the next sequential sample."""

        if action not in (0, 1):
            raise ValueError("action must be 0 or 1")

        true_label = int(self.labels[self.current_index])

        if action == 1 and true_label == 1:
            reward = self.TP_REWARD
            outcome = "TP"

        elif action == 0 and true_label == 0:
            reward = self.TN_REWARD
            outcome = "TN"

        elif action == 1 and true_label == 0:
            reward = self.FP_REWARD
            outcome = "FP"

        else:
            reward = self.FN_REWARD
            outcome = "FN"

        next_index = self.current_index + 1

        done = next_index >= len(self.states)

        if done:
            next_state = self.states[self.current_index]
        else:
            self.current_index = next_index
            next_state = self.states[self.current_index]

        info = {
            "true_label": true_label,
            "action": action,
            "outcome": outcome,
            "index": self.current_index,
        }

        return next_state, reward, done, info


if __name__ == "__main__":
    print("=== Sequential IDS Environment Test ===")

    states = np.array(
        [
            [0.0] * 8,
            [1.0] * 8,
            [2.0] * 8,
        ],
        dtype=np.float32,
    )

    labels = np.array([0, 1, 0], dtype=np.int64)

    env = IDSEnvironment(states, labels)

    state = env.reset(0)

    assert np.all(state == states[0])

    _, reward, done, info = env.step(0)

    assert reward == 0.1
    assert info["outcome"] == "TN"
    assert not done

    _, reward, done, info = env.step(1)

    assert reward == 1.0
    assert info["outcome"] == "TP"
    assert not done

    _, reward, done, info = env.step(1)

    assert reward == -0.5
    assert info["outcome"] == "FP"
    assert done

    print("Sequential transition test: PASSED")
    print("Reward test: PASSED")
    print("IDS environment test: PASSED")
