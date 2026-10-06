# Phase 3: Baseline DQN vs Proposed D3QN

## Objective

Compare the baseline Deep Q-Network (DQN) reproduced in Phase 2 with the proposed Dueling Double Deep Q-Network (D3QN) under the same experimental conditions.

## Controlled Experimental Conditions

The following were kept unchanged between the baseline DQN and proposed D3QN:

- Dataset: UGRansome
- Preprocessing pipeline
- Input features: 8
- Action space: 2
- Reward function
- Training seed: 42
- Training episodes: 10
- Steps per episode: 20,000
- Total training steps: 200,000
- Replay capacity: 100,000
- Batch size: 64
- Learning rate: 0.001
- Discount factor gamma: 0.99
- Epsilon schedule: 1.0 to 0.1
- Epsilon decay: 0.995 per step
- Target network update: end of episode
- Optimizer: Adam
- Loss: MSE
- Evaluation dataset: identical Phase 2 test set
- Evaluation metrics: identical

The primary methodological change is the replacement of the standard DQN architecture and target-value calculation with a D3QN formulation combining:

1. Dueling network architecture
2. Double DQN target selection

## Results

| Metric | Baseline DQN | Proposed D3QN | Change |
|---|---:|---:|---:|
| Accuracy | 0.955695 | 0.961152 | +0.005457 |
| Precision | 0.931244 | 0.942207 | +0.010963 |
| Recall / Detection Rate | 0.993467 | 0.990725 | -0.002742 |
| F1 Score | 0.961350 | 0.965857 | +0.004507 |
| False Positive Rate | 0.091343 | 0.075675 | -0.015668 |
| Specificity | 0.908657 | 0.924325 | +0.015668 |

## Confusion Matrix Comparison

### Baseline DQN

| | Predicted Benign | Predicted Ransomware |
|---|---:|---:|
| Actual Benign | 18,095 | 1,819 |
| Actual Ransomware | 162 | 24,637 |

### Proposed D3QN

| | Predicted Benign | Predicted Ransomware |
|---|---:|---:|
| Actual Benign | 18,407 | 1,507 |
| Actual Ransomware | 230 | 24,569 |

## Observations

The proposed D3QN improves overall accuracy, precision, F1 score, and specificity compared with the reproduced baseline DQN.

The most notable improvement is the reduction in false positives from 1,819 to 1,507, corresponding to 312 fewer false-positive predictions. The false-positive rate decreases from 9.1343% to 7.5675%, while specificity increases from 90.8657% to 92.4325%.

The ransomware detection rate remains very high, decreasing slightly from 99.3467% to 99.0725%. This indicates a small recall trade-off in exchange for substantially fewer false-positive predictions.

## Interpretation

These results provide initial evidence that the D3QN formulation can improve the balance between ransomware detection and false-alarm control under the same experimental setup.

However, this is a single-seed comparison. Statistical significance and robustness across different random seeds have not yet been established. Further experiments are therefore required before making stronger claims about generalization or superiority.

## Phase 3 Status

The D3QN implementation, training, checkpoint generation, and evaluation have been completed successfully.

Next experiments should evaluate reproducibility across multiple random seeds and investigate the individual contributions of the dueling architecture and Double DQN target calculation.
