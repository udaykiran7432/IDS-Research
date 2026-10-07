# Phase 10 Results

Controlled family-held-out evaluation across five seeds.

Values are mean ± sample standard deviation.

| Experiment | Accuracy | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| E1 DQN | 0.9554 ± 0.0045 | 0.9292 ± 0.0072 | 0.9955 ± 0.0008 | 0.9612 ± 0.0037 | 0.0945 ± 0.0105 |
| E2 D3QN | 0.9639 ± 0.0018 | 0.9443 ± 0.0027 | 0.9934 ± 0.0005 | 0.9683 ± 0.0016 | 0.0729 ± 0.0037 |
| E3 D3QN + PER | 0.9642 ± 0.0019 | 0.9442 ± 0.0030 | 0.9942 ± 0.0007 | 0.9685 ± 0.0016 | 0.0732 ± 0.0041 |

## Paired Differences

| Comparison | Metric | Mean Difference | SD |
|---|---|---:|---:|
| E2 D3QN - E1 DQN | Accuracy | +0.008496 | 0.003246 |
| E2 D3QN - E1 DQN | Precision | +0.015143 | 0.005672 |
| E2 D3QN - E1 DQN | Recall | -0.002027 | 0.001064 |
| E2 D3QN - E1 DQN | F1 | +0.007089 | 0.002661 |
| E2 D3QN - E1 DQN | FPR | -0.021593 | 0.008346 |
| E3 D3QN + PER - E2 D3QN | Accuracy | +0.000290 | 0.002255 |
| E3 D3QN + PER - E2 D3QN | Precision | -0.000134 | 0.003658 |
| E3 D3QN + PER - E2 D3QN | Recall | +0.000718 | 0.000829 |
| E3 D3QN + PER - E2 D3QN | F1 | +0.000270 | 0.001922 |
| E3 D3QN + PER - E2 D3QN | FPR | +0.000241 | 0.005079 |
