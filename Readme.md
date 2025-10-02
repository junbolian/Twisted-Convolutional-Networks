You're right — the previous draft read like a quick project note rather than a **formal, citable GitHub README**. Below is a tightened, publication-style README you can drop into `README.md`. It:

* Adds a **projection-based residual** variant with clear **Python & MATLAB** usage (aligned with your MATLAB script).
* States **intended use cases** (many features → prefer projection; accuracy impact is typically small).
* Includes **computational complexity**, **reproducibility**, and a **References** section with canonical works on explicit interaction modeling.
* Keeps your paper as the primary citation and adds BibTeX.

---

# Twisted Convolutional Networks (TCNs)

TCNs explicitly construct interaction features for non-spatial/tabular data and are robust to feature ordering. This repo includes both the original TCN and a **projection-based residual** variant that drastically cuts parameters/VRAM while keeping accuracy close to full-width baselines.

![TCN vs CNN Comparison](TCN.vs.CNN.png)
![TCN Architecture](TCN_Architecture.png)

## Overview

Traditional CNNs depend on adjacency and order; many tabular datasets have neither. **Twisted Convolutional Networks (TCNs)** sidestep this by **explicitly generating interaction features** from subsets of the input dimensions (e.g., multiplicative products or pairwise-product sums). The interaction tensor is then passed to lightweight heads (BN/ReLU/Dropout and residual paths). This repo provides training/evaluation code and a **projection-based residual** that keeps models compact and training stable on wide feature sets.

---

## What’s New — Projection-based Residual (Recommended for Many Features)

Let (F) be the number of original features and (C) the interaction order (default (C=2)). The interaction layer expands to (M=\binom{F}{C}) features. A plain residual block must match this width (costly when (M) is large). The **projection-based residual** inserts a learned linear projection (W_p:\mathbb{R}^M!\to!\mathbb{R}^{H_2}) before the skip addition:

* **Width decoupling**: choose a compact hidden width (H_2 \ll M) without losing a residual pathway.
* **Stability**: projection + normalization keeps post-interaction scales controlled.
* **Practical accuracy**: for typical tabular tasks, projection keeps accuracy close to full-width with far fewer params/VRAM.

> **Guidance**: If (F\ge 20) or memory/latency matters, use **`residual=projection`** with **`H2` in [64, 256]**. For very small (F), a plain residual can serve as an upper-bound baseline.

---

## Features

* **Explicit interactions**: `multiplicative` or `pairwise` construction over feature subsets.
* **Order robustness**: less reliance on any particular feature ordering.
* **Flexible heads**: BN, ReLU, Dropout, and **plain/projection residuals**.
* **Python & MATLAB**: aligned interfaces for reproducible comparisons.

---

## Installation

```bash
git clone https://github.com/junbolian/Twisted_Convolutional_Networks.git
cd Twisted_Convolutional_Networks
pip install -r requirements.txt
```

**Python**: 3.8+ (PyTorch or TensorFlow), NumPy, scikit-learn, Matplotlib
**MATLAB**: R2021b+ recommended (`matlab/` scripts are self-contained)

---

## Quick Start — Python

### Train (projection-based residual)

```bash
python train_tcn.py \
  --dataset iris \
  --epochs 200 \
  --batch_size 16 \
  --combination_method pairwise \
  --residual projection \
  --proj_dim 128 \
  --lr 1e-3 --dropout 0.1 --seed 42
```

**Key args**

* `--combination_method`: `multiplicative` | `pairwise`
* `--order`: interaction order (C) (default 2)
* `--residual`: `none` | `plain` | `projection`
* `--proj_dim`: (H_2) for projection residual (e.g., 64–256)
* Regularization & training: `--bn`, `--dropout`, `--weight_decay`, `--lr`, `--epochs`, `--batch_size`, `--seed`

### Evaluate

```bash
python evaluate_tcn.py \
  --model_path outputs/iris/proj128/best.ckpt \
  --dataset iris
```

---

## Quick Start — MATLAB (Projection-based TCN)

The provided MATLAB script implements the **projection residual** end-to-end and matches this README’s design:

* Reads `dataset.xlsx` (`X = N×F`, labels in the **last** column).
* Randomly **shuffles feature columns** (emphasize order-robustness).
* Builds **pairwise (C=2)** interactions by default.
* Splits with `cvpartition`, trains with **Adam**, and reports accuracy, confusion matrix, ROC/AUC.
* Includes **Input×Gradient** attribution with human-readable combination names (mapped back to original feature indices).

**Key editable knobs in the script**

* `combination_method = 'pairwise'` or `'multiplicative'`
* `num_combinations = 2`  (interaction order (C))
* `H1 = 64; H2 = 256;`  (set `H2` to 64–256 for compact projection)
* `residual_source = 'input';`  (skip source; `'relu1'` is also supported)
* Training options (epochs, batch size, LR, weight decay)

> See the top-level MATLAB example in your repo. It already includes: training curves, confusion matrix, per-class Precision/Recall/F1, ROC/AUC, and Top-K interaction attributions.

---

## When Should I Use Projection?

| Scenario                                                    | Suggested Setup                                        | Why                                |
| ----------------------------------------------------------- | ------------------------------------------------------ | ---------------------------------- |
| **Many features** ((F \ge 20)) or higher-order interactions | `residual=projection`, `proj_dim∈[64,256]`, `pairwise` | Cuts params/VRAM, stable scales    |
| **Small (F)** or quick ablations                            | `residual=plain` or `none`                             | Simpler upper-bound baseline       |
| **Overfitting risk**                                        | `pairwise` + BN + Dropout + projection                 | Smoother features + regularization |
| **Latency/VRAM constraints**                                | Lower `proj_dim`, prefer `multiplicative`              | Fewer params, faster I/O           |

> In practice, if accuracy dips with projection, bump `proj_dim` (e.g., 128 → 192/256) before changing `order`.

---

## Computational Complexity

* **Interaction size**: (M=\binom{F}{C}).
* **Time**: interaction construction (O(M)) per sample; head scales (O(M\cdot H_2)) with projection vs. (O(M^2)) for a full-width residual ((H_2=M)).
* **Memory**: activations/params scale with (M). **Projection** reduces to (O(M\cdot H_2)) with (H_2!\ll!M).

---

## Reproducibility & Versioning

* Default **seed=42** (Python & MATLAB examples).
* Pinned `requirements.txt` for Python.
* This README and MATLAB script target **TCN v1.1** (Last update: **Sep 29, 2025**).

---

## File Map

```
.
├─ train_tcn.py                # Training entry (Python)
├─ evaluate_tcn.py             # Evaluation entry (Python)
├─ TCNs_projection_based.py    # Projection residual reference (Python)
├─ matlab/
│  └─ tcn_projection_demo.m    # End-to-end script (projection residual; matches README)
├─ notebooks/                  # Demos (Iris, Breast Cancer, custom CSVs)
├─ results/                    # Logs, metrics, plots
├─ requirements.txt
└─ README.md
```

---

## Results (Brief)

Across common tabular benchmarks where feature order is arbitrary, TCNs are competitive with MLP-style baselines and avoid the order sensitivity of CNNs. The **projection-based residual** typically matches full-width accuracy with a fraction of parameters. See `results/` or the paper for full metrics and ablations.

---

## Citation

If you use this code, please cite:

```bibtex
@article{lian2026tcns,
  title={Twisted Convolutional Networks (TCNs): Enhancing Feature Interactions for Non-Spatial Data Classification},
  author={Junbo Jacob Lian, Haoran Chen, Kaichen Ouyang, Yujun Zhang, Rui Zhong, Huiling Chen},
  journal={Neural Networks},
  year={2026}
}
```

---

## License

Released under the **MIT License**. See [LICENSE](LICENSE).

---

## Contact

**Junbo Lian** — [jacoblian@u.northwestern.edu](mailto:jacoblian@u.northwestern.edu)
