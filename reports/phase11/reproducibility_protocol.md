# Phase 11 — Reproducibility Protocol

## 1. Purpose

This document records the frozen environment, dataset construction,
experimental configurations, evaluation procedure, and verification checks
required to reproduce the final IDS experiments.

Phase 10 results are treated as frozen research results. Phase 11 does not
introduce new model training, hyperparameter tuning, or result modification.

## 2. Dataset

Source dataset:

- UGRansome
- Raw file: `data/raw/UGRansome.csv`

The family-held-out evaluation uses Locky as the unseen attack family.

## 3. Family-held-out split

Random state:

- `42`

Known-family split:

- 70% known-family data → known training pool
- 30% known-family data → final test portion

Unseen-family handling:

- All Locky samples are excluded before the known-family split.
- All Locky samples are placed in the final test set.
- Locky is absent from model training and calibration.

Known training pool:

- 80% → model training
- 20% → calibration

The training/calibration split is stratified jointly by family and original
target.

## 4. Preprocessing

Features:

1. Time
2. Protcol
3. Flag
4. Family
5. Clusters
6. Threats
7. USD
8. BTC

Categorical features:

- Protcol
- Flag
- Family
- Threats

Numerical features:

- Time
- Clusters
- USD
- BTC

Categorical preprocessing:

- `OrdinalEncoder`
- fitted only on model-training data
- unknown categories encoded as `-1`

Numerical preprocessing:

- `MinMaxScaler`
- fitted only on model-training data

Binary target:

- `S` → 0 (benign)
- `A` → 1 (ransomware)
- `SS` → 1 (ransomware)

## 5. Frozen dataset sizes

| Split | Samples |
|---|---:|
| Training | 69,428 |
| Calibration | 17,358 |
| Test | 62,257 |
| Unseen Locky test samples | 25,062 |

Training target counts:

- benign: 30,929
- ransomware: 38,499

Calibration target counts:

- benign: 7,732
- ransomware: 9,626

Test target counts:

- benign: 27,719
- ransomware: 34,538

Leakage checks:

- Locky in training: 0
- Locky in calibration: 0
- Locky in test: 25,062

## 6. Controlled experiments

### E1 — DQN

- DQN architecture
- uniform replay
- standard DQN target

### E2 — D3QN

- D3QN architecture
- uniform replay
- Double-DQN target

### E3 — D3QN + PER

- D3QN architecture
- prioritized experience replay
- Double-DQN target

Common controlled settings:

- state dimension: 8
- action dimension: 2
- learning rate: 0.001
- gamma: 0.99
- epsilon start: 1.0
- epsilon end: 0.1
- epsilon decay: 0.995
- batch size: 64
- replay capacity: 100,000
- device: CPU

PER settings:

- alpha: 0.6
- beta start: 0.4
- beta increment: 0.001

Training:

- 10 episodes
- maximum 20,000 steps per episode

## 7. Controlled multi-seed evaluation

Phase 10 evaluates E1, E2, and E3 using five seeds:

- 42
- 7
- 21
- 123
- 2026

Each experiment produces a checkpoint and corresponding training/evaluation
reports.

## 8. Closed-set evaluation

E1–E3 are closed-set classifiers.

For known-family evaluation:

- predictions are obtained from the policy network
- action 0 represents benign
- action 1 represents ransomware

For Locky:

- no UNKNOWN rejection mechanism exists in E1–E3
- therefore every Locky sample is necessarily assigned to a known class
- Locky UNKNOWN detection rate is 0
- Locky false-accept rate is 1

Known-family classification metrics include:

- accuracy
- precision
- recall
- F1
- false-positive rate
- specificity
- confusion matrix

## 9. Reproducibility verification

The following checks were performed without modifying the frozen Phase 10
results:

1. Phase 10 Git working tree checked.
2. Phase 10 tracked files showed no modifications.
3. Phase 10 checkpoint modification status checked.
4. Phase 7 processed dataset row counts verified.
5. Dataset schemas verified.
6. Family and target distributions verified.
7. Locky leakage checks verified.
8. Phase 10 summary/statistical/manifest files verified to exist and parse.
9. Environment versions recorded in `environment.json`.
10. Experiment source-to-output mappings recorded in `experiment_manifest.json`.

## 10. Frozen research state

Phase 10 is the frozen experimental result state.

Phase 11 is documentation, reproducibility verification, and finalization.
It must not alter previously accepted Phase 10 results.

Any future experiment, tuning, alternative split, or methodological change
must be treated as a new experimental version rather than silently replacing
the frozen results.
