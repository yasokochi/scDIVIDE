# scBURST

scBURST models developmental trajectories with growth-dependent diffusion stochastic differential equations and unbalanced optimal transport.

> Cite: hogehoge

---

## 1. Install scBURST

GitHub repository: `scBURST`

Clone the repository and install it in editable/development mode:

```bash
git clone <your-github-url>/scBURST.git
cd scBURST
pip install -e .
```

> Important: `scBURST` is built on **PyTorch**, **TorchSDE**, and **POT**.
> Install `torch`, `torchsde`, and `pot` yourself first for your environment before installing or running `scBURST`.

---

## 2. Install dependencies first

Install the required packages before running `scBURST`:

```bash
pip install torch torchsde pot
```

`scBURST` uses:
- `torch` for neural network training
- `torchsde` for stochastic differential equation integration
- `pot` for optimal transport computations

---

## 3. Basic Usage Example

```python
import torch
from scburst import BranchingSDE, train_loop

func = BranchingSDE(
    in_out_dim=3,
    sigma=0.05,
    delta=0.1,
    alpha=0.5,
    r0=1.0,
)

results = train_loop(
    func=func,
    data_train=data_train,
    train_time=train_time,
    integral_time=integral_time,
    device=device,
    train_config=train_config,
    sde_config=sde_config,
    sinkhorn_config=sinkhorn_config,
    optimizer_config=optimizer_config,
    scheduler_config=scheduler_config,
    reg_config=reg_config,
    output_config=output_config,
)
```

> For a complete runnable example, see [examples](examples/).

---

## 4. Example Notebook

A bundled notebook example based on the synthetic three-gene dataset with `alpha = 0.5` is included in this repository:

- `examples/three_gene_alpha05.ipynb`
- `examples/data/three_gene_scburst_alpha05_data.csv`

Open the notebook from the repository root and run the cells in order.

---

## 5. Runtime Reference

Approximate `scBURST` training time for 500 steps on a single NVIDIA A6000 GPU:

| Dataset | Time for 500 steps |
|---------|-------------------:|
| Three-gene (3D) | 790 s |
| Mouse hematopoiesis (50D) | 260 s |

These values are rough references derived from the supplementary benchmark by scaling the reported per-100-step runtime by 5.
In that benchmark, `scBURST` used 400 subsampled particles per iteration.
With holdout index `k = 1`, the three-gene dataset retained 3 training intervals and the mouse hematopoiesis dataset retained 1 training interval.

---

## 6. Summary

| Item | Value |
|------|-------|
| Core framework | PyTorch |
| SDE solver | TorchSDE |
| OT library | POT |
| Example | `examples/three_gene_alpha05.ipynb` |
| GitHub repository | `scBURST` |

---

## 7. Citation

If you use `scBURST` in your work, please cite:

```text
hogehoge
```
