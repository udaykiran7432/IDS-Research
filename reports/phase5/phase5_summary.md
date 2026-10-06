# Phase 5 — D3QN + PER Five-Seed Analysis

## Experiment

Five-seed controlled evaluation of DQN, D3QN, and D3QN + Prioritized
Experience Replay (PER).

Seeds: [42, 7, 21, 123, 2026]

Test samples: 44,713

## Aggregate Results

| Metric | DQN | D3QN | D3QN + PER |
|---|---:|---:|---:|
| Accuracy | 0.952206 ± 0.012376 | 0.955901 ± 0.009824 | 0.962516 ± 0.000849 |
| Precision | 0.925340 ± 0.020021 | 0.931513 ± 0.016781 | 0.943208 ± 0.001763 |
| Recall | 0.994516 ± 0.002324 | 0.993871 ± 0.002331 | 0.992161 ± 0.001789 |
| F1 | 0.958575 ± 0.010087 | 0.961605 ± 0.008118 | 0.967063 ± 0.000738 |
| FPR | 0.100482 ± 0.029970 | 0.091383 ± 0.024404 | 0.074400 ± 0.002548 |
| Specificity | 0.899518 ± 0.029970 | 0.908617 ± 0.024404 | 0.925600 ± 0.002548 |

## D3QN → D3QN + PER Mean Changes

- Accuracy: +0.006616
- Precision: +0.011695
- Recall: -0.001710
- F1: +0.005459
- FPR: -0.016983
- Specificity: +0.016983

## G1 Conclusion

**Supported with qualification.**

D3QN + PER improves overall performance and false-positive control
relative to D3QN across the five controlled seeds while maintaining
very high ransomware detection recall.

The small recall reduction is explicitly reported.

No statistical-significance claim is made from the current five-seed
experiment.

## Paired Win Counts

- Accuracy: PER 4/5
- Precision: PER 4/5
- Recall: PER 1/5
- F1: PER 4/5
- FPR: PER 4/5
- Specificity: PER 4/5
