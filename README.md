# Adaptive Open-Set Intrusion Detection System

**Final-Year Research Project — Intrusion Detection, Deep Reinforcement Learning, Open-Set Recognition, and Explainable AI**

> **Current research state:** Phases 0–11 are completed and reproducibility artifacts are frozen. Phase 12 is the research-paper stage.

---

## 1. Project Overview

This repository contains the implementation and experimental evaluation of an adaptive network intrusion detection system (IDS) designed to address a central limitation of conventional closed-set intrusion detection:

> **Can an adaptive IDS identify network traffic from an attack family that was unseen during training as `UNKNOWN`, instead of incorrectly forcing it into a known attack class, while maintaining strong detection performance and providing interpretable evidence for its decisions?**

The project starts from a DQN-based adaptive IDS baseline and develops a controlled research pipeline through:

```text
DQN
  ↓
D3QN
  ↓
D3QN + Prioritized Experience Replay (PER)
  ↓
VAE learned representation
  ↓
Open-set recognition
  ↓
Integrated IDS
  ↓
SHAP explainability
```

The research is evaluated using a **family-held-out** protocol in which an attack family is deliberately excluded from model training and introduced only during final testing as an unseen family.

---

# 2. Research Motivation

A conventional closed-set classifier assumes that test samples belong to classes represented during training.

That assumption is problematic for operational cybersecurity because new attack families can appear after deployment. If an unseen attack resembles a known attack, a closed-set IDS can produce a confident but incorrect known-class prediction.

This project therefore investigates an IDS that can distinguish:

```text
KNOWN
  ├── Normal
  └── Known attack

UNKNOWN
  └── Attack family not observed during training
```

The project additionally investigates whether learned representations and explainability can make such a system more useful for security analysis.

---

# 3. Research Gaps

The research is organized around four gaps.

## G1 — Reinforcement-learning learning efficiency

The DQN baseline provides the starting adaptive learning mechanism. The project investigates whether improvements in value estimation and experience selection provide better learning behavior.

```text
DQN
 ↓
D3QN
 ↓
D3QN + PER
```

**Research component:** D3QN + Prioritized Experience Replay.

---

## G2 — Feature representation

A manually/preprocessed flow representation may not provide the most useful representation for downstream detection.

The project investigates a Variational Autoencoder (VAE) that produces:

```text
Input flow
   ↓
VAE encoder
   ↓
Latent representation
   +
Reconstruction error
```

The latent representation and reconstruction error are then available to the downstream IDS.

---

## G3 — Closed-set assumption

A conventional classifier can force an unseen attack into one of its known classes.

The project therefore evaluates an open-set mechanism:

```text
Model score / representation evidence
              ↓
       Open-set decision
          ↙       ↘
      KNOWN      UNKNOWN
```

The unseen family is never used for model training.

---

## G4 — Explainability

Security analysts need evidence for model decisions.

SHAP is therefore used to explain representative model outputs, including correct decisions, false accepts, and misclassifications.

SHAP is treated as an explanation mechanism; the project does not claim that SHAP itself improves detection accuracy.

---

# 4. Research Question

The central research question is:

> **Can an adaptive intrusion detection system identify network traffic from attack families unseen during training as “unknown,” rather than incorrectly forcing them into known attack classes, while maintaining good detection performance and providing interpretable evidence for decisions?**

The project is structured so that each major research component can be evaluated independently before being combined.

---

# 5. Research Chain

The research follows this progression:

```text
Better RL learning
        ↓
Better representation
        ↓
Unknown-attack detection
        ↓
Interpretable decisions
```

The target integrated architecture is:

```text
                UGRansome
                    │
                    ▼
             Preprocessing
                    │
                    ▼
                  VAE
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   Latent vector       Reconstruction
                            error
          └─────────┬─────────┘
                    ▼
              D3QN + PER
                    │
                    ▼
           Open-set decision
              ↙         ↘
          KNOWN        UNKNOWN
              \         /
               \       /
                ▼     ▼
                   SHAP
                    │
                    ▼
          IDS decision + evidence
```

---

# 6. Dataset and Experimental Protocol

The project uses the **UGRansome** network-flow dataset.

The most important evaluation design is **family-held-out evaluation**.

## Unseen-family protocol

The project selects **Locky** as the unseen attack family.

Conceptually:

```text
                 UGRansome
                     │
             ┌───────┴────────┐
             │                │
       Known families        Locky
             │                │
             ▼                │
       Train / Calibration   │
                              │
                              ▼
                         Final Test only
```

Locky is excluded from model training and calibration and appears in the final test set as the unseen family.

This is intended to evaluate the open-set research question rather than merely testing attacks from families already observed during training.

## Recorded split

The Phase 7/11 reproducibility record contains:

| Split | Rows |
|---|---:|
| Model training | 69,428 |
| Calibration | 17,358 |
| Final test | 62,257 |

The final test set contains:

- **37,195** known-family samples
- **25,062** Locky samples

The preprocessing protocol is documented in the repository's Phase 11 reproducibility record.

---

# 7. Data Leakage Controls

The experimental protocol explicitly addresses leakage.

The recorded procedure includes:

- Locky excluded from model training;
- Locky excluded from calibration;
- preprocessing fitted only on model-training data;
- calibration separated from model training;
- final held-out test set preserved for evaluation;
- family information retained for family-held-out analysis.

The split and preprocessing procedure is implemented in:

```text
src/open_set/create_split.py
```

The reproducibility documentation is:

```text
reports/phase11/reproducibility_protocol.md
```

---

# 8. Experimental Ladder

The project uses a controlled experimental ladder.

| Experiment | Configuration | Research purpose |
|---|---|---|
| **E1** | DQN | Baseline |
| **E2** | D3QN | G1 |
| **E3** | D3QN + PER | G1 |
| **E4** | D3QN + PER + VAE | G2 |
| **E5** | D3QN + PER + Open-set | G3 |
| **E6** | D3QN + PER + VAE + Open-set | Integrated system |
| **E7** | Full system + SHAP | G4 / explainability |

The experimental ladder is deliberately incremental so that individual components can be evaluated before conclusions are drawn about the integrated system.

---

# 9. Research Development Phases

The project follows a locked 12-phase research plan.

| Phase | Research activity | Status |
|---|---|---|
| **0** | Project foundation | ✅ Complete |
| **1** | Research definition | ✅ Complete |
| **2** | Dataset/data audit | ✅ Complete |
| **3** | Preprocessing + family-held-out split | ✅ Complete |
| **4** | DQN baseline | ✅ Complete |
| **5** | D3QN + PER | ✅ Complete |
| **6** | VAE representation learning | ✅ Complete |
| **7** | Open-set mechanism | ✅ Complete |
| **8** | Integrated model | ✅ Complete |
| **9** | SHAP explainability | ✅ Complete |
| **10** | Evaluation + ablation | ✅ Complete |
| **11** | Reproducibility + finalization | ✅ Complete |
| **12** | Research paper | 🔄 Current / next |

Each implementation phase followed the same research discipline:

```text
IMPLEMENT
   ↓
TEST
   ↓
VALIDATE
   ↓
SAVE RESULTS
   ↓
GIT COMMIT
   ↓
NEXT PHASE
```

---

# 10. Controlled Multi-Seed Evaluation

The controlled E1–E3 comparison uses five seeds:

```text
42
7
21
123
2026
```

The purpose is to avoid basing the main DQN/D3QN/PER comparison on a single random initialization.

The saved Phase 10 artifacts include the corresponding checkpoints, training reports, evaluation reports, paired differences, statistical analysis, summary tables, and figures.

---

# 11. Main Controlled Results

The Phase 10 controlled results are reported as mean ± standard deviation across five seeds.

## DQN vs D3QN

| Metric | DQN | D3QN |
|---|---:|---:|
| Accuracy | 0.955386 ± 0.004478 | **0.963882 ± 0.001833** |
| Precision | 0.929180 ± 0.007200 | **0.944323 ± 0.002659** |
| Recall | **0.995472 ± 0.000780** | 0.993445 ± 0.000533 |
| F1 | 0.961171 ± 0.003718 | **0.968260 ± 0.001571** |
| FPR | 0.094508 ± 0.010460 | **0.072915 ± 0.003664** |

### Interpretation

Across the five seeds, D3QN consistently improved overall classification metrics and reduced false-positive rate relative to DQN, while showing a small reduction in recall.

The paired analysis found improvements in several metrics, but because only five seeds were used, the non-parametric results are interpreted conservatively. The project therefore avoids presenting these results as universal statistical proof.

---

# 12. D3QN vs D3QN + PER

| Metric | D3QN | D3QN + PER |
|---|---:|---:|
| Accuracy | 0.963882 | 0.964173 |
| Precision | 0.944323 | 0.944189 |
| Recall | 0.993445 | 0.994162 |
| F1 | 0.968260 | 0.968529 |
| FPR | 0.072915 | 0.073156 |

### Interpretation

The additional effect of PER was small and inconsistent across metrics.

The paired statistical analysis did not show a paired p-value below 0.05 for the E3-versus-E2 comparisons.

Therefore:

> **The current experiments do not justify claiming that PER provides a large or statistically established additional improvement over D3QN.**

This is retained as an experimental finding and limitation rather than hidden from the research narrative.

---

# 13. VAE Representation Learning

The VAE introduces a learned representation consisting of:

```text
latent vector
+
reconstruction error
```

The integrated pipeline uses this representation before the downstream D3QN/PER decision process.

The VAE experiments are documented under:

```text
src/vae/
src/integrated/
reports/phase6/
reports/phase8/
```

The project also performs a controlled **no-Family VAE ablation** in Phase 10.

---

# 14. VAE No-Family Ablation

The no-Family ablation is important because it tests whether the observed open-set separation is robust when the `Family` feature is removed.

Recorded Phase 10 findings include:

```text
VAE input features: 7
Hidden dimension: 16
Latent dimension: 4
Seed: 42
Calibration threshold: 0.2319232374
```

For unseen Locky traffic:

```text
Locky false-accept rate ≈ 95.41%
```

The downstream no-Family E4b evaluation also produced:

```text
Known closed-set accuracy ≈ 91.36%
Known closed-set F1 ≈ 90.78%
```

### Interpretation

The no-Family experiment indicates that the observed open-set separation depends substantially on the available feature representation.

Therefore the project **does not claim that the VAE alone provides robust unseen-family detection independent of feature choice**.

This is an important methodological limitation and an explicit part of the research discussion.

---

# 15. Open-Set Recognition

The project compares a closed-set decision against an open-set decision.

## Closed-set behavior

Every sample must be assigned to a known class:

```text
Locky
  ↓
forced into known class
```

## Open-set behavior

The system can reject a sample:

```text
Locky
  ↓
open-set gate
  ↓
UNKNOWN
```

---

# 16. Open-Set Ablation Results

The Phase 10 gate ablation uses frozen E6 predictions.

Recorded results include:

| Measure | Result |
|---|---:|
| Known closed-set accuracy | 0.920043 |
| Known closed-set precision | 0.880721 |
| Known closed-set recall | 0.989867 |
| Known closed-set F1 | 0.932110 |
| Known closed-set FPR | 0.166868 |
| Open-set rejection of known samples | 0.052077 |
| Locky samples | 25,062 |
| Locky detected as UNKNOWN | 14,635 |
| Locky unknown-detection rate | 0.583952 |
| Locky false-accept rate | 0.416048 |

### Interpretation

The open-set mechanism provides a genuine rejection path for unseen traffic, but it does not reject all unseen Locky traffic.

Therefore the result supports the presence of an open-set capability while also demonstrating a substantial false-accept limitation.

---

# 17. SHAP Explainability

SHAP is used to provide feature-level explanations for representative model decisions.

The Phase 9 analysis contains representative cases covering:

- known benign traffic correctly classified;
- known ransomware correctly classified;
- unseen Locky correctly identified as unknown;
- Locky false acceptance;
- known misclassification.

For the analyzed D3QN ransomware action score, the largest average absolute SHAP contributions included:

```text
reconstruction error
latent feature 3
latent feature 2
latent feature 0
latent feature 1
```

The corresponding explanation artifacts are stored under:

```text
reports/phase9/
```

### Important interpretation

SHAP is used here to explain model outputs.

It is **not** treated as proof of causal feature importance and is **not** claimed to improve model accuracy.

---

# 18. Evaluation Metrics

The project evaluates multiple aspects of IDS behavior.

## Classification metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Security metrics

- False-positive rate (FPR)
- False-negative rate (FNR)
- Known attack recall
- Unknown detection rate
- Unknown recall
- Known-vs-unknown behavior

## Ranking metrics

Where appropriate:

- ROC-AUC
- PR-AUC

## Reinforcement-learning metrics

- Reward
- Loss
- Convergence behavior
- Training time

## VAE metrics

- Reconstruction loss
- Reconstruction-error distribution
- Latent representation behavior

## System metrics

- Latency
- Throughput
- Model size
- Memory requirements

## Explainability

Representative explanations for:

- correct known decisions;
- correct unknown decisions;
- false accepts;
- misclassifications.

---

# 19. Ablation Strategy

The ablation structure directly corresponds to the research gaps.

## G1 — Reinforcement learning

```text
DQN
 ↓
D3QN
 ↓
D3QN + PER
```

## G2 — Representation

```text
Original/preprocessed representation
              ↓
          VAE representation
```

## G3 — Open-set recognition

```text
Closed-set
    ↓
Open-set
```

## G4 — Explainability

```text
Model output
    ↓
SHAP explanation
```

The project deliberately avoids assuming improvement before the experiments are performed.

---

# 20. Repository Structure

The repository is organized around research components and experimental stages.

```text
IDS-Research/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── artifacts/
│   └── phase10/
│
├── reports/
│   ├── phase3/
│   ├── phase4/
│   ├── phase5/
│   ├── phase6/
│   ├── phase7/
│   ├── phase8/
│   ├── phase9/
│   ├── phase10/
│   └── phase11/
│
├── src/
│   ├── baseline/
│   ├── proposed/
│   ├── vae/
│   ├── open_set/
│   ├── integrated/
│   ├── shap_explain/
│   └── evaluation/
│
├── .gitignore
└── README.md
```

### `src/baseline/`

DQN baseline training and evaluation.

### `src/proposed/`

D3QN-based proposed reinforcement-learning implementation.

### `src/vae/`

VAE model, training, and validation.

### `src/open_set/`

Family-held-out splitting, calibration, threshold analysis, open-set training, and evaluation.

### `src/integrated/`

VAE + D3QN/PER + open-set integration.

### `src/shap_explain/`

Global and representative-case SHAP analysis.

### `src/evaluation/`

Controlled multi-seed experiments, E4b ablation, open-set ablation, statistical analysis, and Phase 10 output generation.

### `reports/`

Saved experimental results, figures, statistical analyses, and reproducibility documentation.

### `artifacts/`

Selected trained model checkpoints required for reproducibility of the controlled experiments.

---

# 21. Reproducibility

Phase 11 froze the reproducibility state of the project.

The repository contains:

```text
reports/phase11/environment.json
reports/phase11/experiment_manifest.json
reports/phase11/reproducibility_protocol.md
```

## Recorded environment

```text
Python        3.12.10
Platform      macOS / arm64

NumPy         2.5.3
Pandas        3.0.6
SciPy         1.18.1
scikit-learn  1.9.1
PyTorch       2.14.1
SHAP          0.52.0
Matplotlib    3.11.2
ipykernel     7.4.0
JupyterLab    4.6.4
imbalanced-learn 0.14.2
```

The recorded Python executable is the project virtual environment.

## Reproducibility artifacts

Phase 10 retains:

- five E1 checkpoints per seed;
- five E2 checkpoints per seed;
- five E3 checkpoints per seed;
- E4b checkpoint;
- no-Family VAE checkpoint;
- training reports;
- evaluation reports;
- statistical analysis;
- paired differences;
- summary tables;
- figures.

Phase 11 records the experiment manifest and reproduction protocol.

---

# 22. Reproducibility Principle

The project follows:

```text
Dataset
   ↓
Controlled split
   ↓
Training-only preprocessing
   ↓
Model training
   ↓
Calibration
   ↓
Held-out evaluation
   ↓
Saved result
   ↓
Git commit
```

The final paper should only make claims that can be traced to these saved artifacts and documented procedures.

---

# 23. Git and Version Control

Git is used to preserve research checkpoints and prevent accidental loss of experimental states.

The current repository contains these phase branches:

```text
main
phase3-proposed-method
phase4-experimental-evaluation
phase6-vae
phase7-open-set
phase8-integrated-model
phase9-shap
phase10-evaluation
```

The historical branch names reflect how the project evolved during development; they are preserved rather than rewritten.

## Current authoritative branch

```text
main
```

## Current validated checkpoint

```text
28021af
Complete Phase 11 reproducibility and finalization
```

At this checkpoint:

```text
local main          → 28021af
origin/main         → 28021af
phase10-evaluation  → 28021af
origin/phase10-evaluation → 28021af
```

The remaining historical phase branches are also published to the remote repository.

---

# 24. Research Development Workflow

For future research changes:

```text
main
  │
  ├── create dedicated phase/research branch
  │
  ▼
implementation
  │
  ▼
tests
  │
  ▼
validation
  │
  ▼
results
  │
  ▼
Git commit
  │
  ▼
GitHub push
  │
  ▼
review
  │
  ▼
merge into main
```

This workflow is intended to keep the research reproducible and recoverable.

---

# 25. Important Research Findings

The current results support several carefully bounded conclusions.

### Finding 1 — D3QN improved the controlled baseline

D3QN produced higher mean accuracy and F1 and lower mean FPR than DQN across the five controlled seeds.

### Finding 2 — PER's additional effect was small

D3QN + PER produced only marginal differences relative to D3QN, with no paired p-value below 0.05 in the Phase 10 comparison.

### Finding 3 — Open-set rejection is useful but imperfect

The open-set gate rejected a substantial fraction of unseen Locky traffic, but a substantial fraction was still accepted as known.

### Finding 4 — Representation choice matters

The no-Family VAE ablation produced substantially poorer unseen-family separation, showing that open-set performance is dependent on the representation and feature configuration.

### Finding 5 — Explainability can expose decision evidence

SHAP provides feature-level evidence for representative decisions, including correct predictions and failure cases.

These findings are the basis for the research-paper discussion; they are not presented as claims of universal superiority.

---

# 26. Limitations

The current research has several limitations.

1. The main controlled DQN/D3QN/PER comparison uses five random seeds.
2. PER did not demonstrate a large additional improvement over D3QN.
3. Open-set recognition does not reject every unseen Locky sample.
4. The no-Family VAE ablation shows strong dependence on feature representation.
5. SHAP explains selected model outputs but should not be interpreted as causal proof.
6. The experiments use the UGRansome dataset and the defined family-held-out protocol; results should not automatically be generalized to every network environment.
7. The current evidence should be interpreted within the tested experimental configuration and should not be presented as proof that every component improves every metric.

These limitations are retained intentionally for scientific transparency.

---

# 27. Research Contribution

The project investigates an integrated IDS pipeline combining:

```text
Adaptive deep reinforcement learning
              +
Learned flow representation
              +
Open-set recognition
              +
Explainability
```

The central research contribution is an **experimental investigation of these components under a family-held-out unseen-attack setting**, with controlled baselines, multi-seed evaluation, ablations, statistical analysis, and reproducibility artifacts.

The contribution is therefore framed around the experimental evidence rather than claiming that every proposed component independently improves the system.

---

# 28. Current Research Status

| Area | Status |
|---|---|
| Project foundation | ✅ Complete |
| Research definition | ✅ Complete |
| Dataset audit | ✅ Complete |
| Preprocessing | ✅ Complete |
| DQN baseline | ✅ Complete |
| D3QN | ✅ Complete |
| D3QN + PER | ✅ Complete |
| VAE | ✅ Complete |
| Open-set mechanism | ✅ Complete |
| Integrated model | ✅ Complete |
| SHAP | ✅ Complete |
| Controlled evaluation | ✅ Complete |
| Ablations | ✅ Complete |
| Statistical analysis | ✅ Complete |
| Reproducibility | ✅ Complete |
| Git/GitHub synchronization | ✅ Complete |
| Research paper | 🔄 Phase 12 |

---

# 29. Phase 12 — Research Paper

The next stage is to convert the completed experimental record into a research paper.

The paper will be built from:

```text
Research question
      ↓
Research gaps
      ↓
Methodology
      ↓
Experimental design
      ↓
Controlled experiments
      ↓
Ablations
      ↓
Statistical analysis
      ↓
Discussion
      ↓
Limitations
      ↓
Conclusion
```

The planned paper structure is:

1. **Introduction**
2. **Related Work**
3. **Problem Formulation**
4. **Methodology**
5. **Experimental Setup**
6. **Results**
7. **Ablation and Statistical Analysis**
8. **Explainability Analysis**
9. **Discussion**
10. **Limitations**
11. **Conclusion**
12. **References**

The paper will distinguish clearly between:

- measured results;
- interpretation;
- limitations;
- future work.

No unsupported performance claim should be introduced during paper writing.

---

# 30. Citation

The formal citation for this work will be added after the research paper and publication details are finalized.

---

# 31. License

A project license will be selected before the final public release of the repository.

---

# 32. Acknowledgement

This repository contains the implementation, experimental evaluation, analysis, and reproducibility artifacts developed as part of a final-year research project investigating adaptive and open-set intrusion detection.

---

## Project Status

**Implementation:** Complete  
**Experimental evaluation:** Complete  
**Ablation analysis:** Complete  
**Statistical analysis:** Complete  
**Reproducibility:** Complete  
**GitHub version control:** Complete  
**Research paper:** Phase 12 — In progress
