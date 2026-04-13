# scDIVIDE

scDIVIDE models developmental trajectories with growth-dependent diffusion stochastic differential equations and unbalanced optimal transport.

> Cite: hogehoge

---

## 1. Install scDIVIDE

GitHub repository: `scDIVIDE`

Clone the repository and install it in editable/development mode:

```bash
git clone <your-github-url>/scDIVIDE.git
cd scDIVIDE
pip install -e .
```

> Important: `scDIVIDE` is built on **PyTorch**, **TorchSDE**, and **POT**.
> Install `torch`, `torchsde`, and `pot` yourself first for your environment before installing or running `scDIVIDE`.

---

## 2. Install dependencies first

Install the required packages before running `scDIVIDE`:

```bash
pip install torch torchsde POT
```

`scDIVIDE` uses:
- `torch` for neural network training
- `torchsde` for stochastic differential equation integration
- `pot` for optimal transport computations

---

## 3. Basic Usage Example

```python
import torch
from scdivide import BranchingSDE, train_loop

func = BranchingSDE(
    in_out_dim=3,
    sigma=0.05,
    delta=0.1,
    alpha=0.5,
).to(device)

results = train_loop(
    func=func,
    data_train=data_train,
    train_time=train_time,
    integral_time=integral_time,
    device=device,
    train_config={"niters": 500, "num_samples": 400},
    optimizer_config={"lr": 1e-3},
)
```

Only `train_config` is required. SDE, Sinkhorn, optimizer, scheduler, regularizer, and output configs are all optional and fall back to sensible defaults. See [`docs/configs.md`](docs/configs.md) for the full list of options.

> Note: the built-in defaults are tuned for the **mouse hematopoiesis** dataset. For the three-gene dataset, please use the hyperparameters reported in the manuscript.

> For a complete runnable example, see [examples](examples/).

---

## 4. Example Notebook

A bundled notebook example based on the synthetic three-gene dataset with `alpha = 0.5` is included in this repository:

- `examples/three_gene_alpha05.ipynb`
- `examples/data/three_gene_scdivide_alpha05_data.csv`

Open the notebook from the repository root and run the cells in order.

---

## 5. Runtime Reference

Approximate `scDIVIDE` training time for 500 steps on a single NVIDIA A6000 GPU:

| Dataset | Time for 500 steps |
|---------|-------------------:|
| Three-gene (3D) | 790 s |
| Mouse hematopoiesis (50D) | 260 s |

These values are rough references derived from the supplementary benchmark by scaling the reported per-100-step runtime by 5.
In that benchmark, `scDIVIDE` used 400 subsampled particles per iteration.
With holdout index `k = 1`, the three-gene dataset retained 3 training intervals and the mouse hematopoiesis dataset retained 1 training interval.

---

## 6. Summary

| Item | Value |
|------|-------|
| Core framework | PyTorch |
| SDE solver | TorchSDE |
| OT library | POT |
| Example | `examples/three_gene_alpha05.ipynb` |
| GitHub repository | `scDIVIDE` |

---

## 7. Citation

If you use `scDIVIDE` in your work, please cite:

```text
hogehoge
```
