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

> Important: `scDIVIDE` is built on **[PyTorch](https://pytorch.org/)**, **[torchsde](https://github.com/google-research/torchsde)**, and **[POT](https://pythonot.github.io/)**.
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
from scdivide import NeuralSDE, train_loop

func = NeuralSDE(
    in_out_dim=3,
    sigma=0.05,
    delta=0.1,
    alpha=0.5,
).to(device)

results = train_loop(
    func=func,
    data_train=data_train,
    integral_time=integral_time,
    device=device,
    train_config={"niters": 500, "num_samples": 400},
    optimizer_config={"lr": 1e-3},
)
```

See [`docs/configs.md`](docs/configs.md) for the full list of options.

> **Important:** scDIVIDE uses Sinkhorn divergence as the fitting loss. Please make sure the Sinkhorn iterations converge and adjust `sinkhorn_config` (e.g., `epsilon`, `n_iter`) if needed.
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
| Three-gene (3D, 5 timepoints) | 790 s |
| Mouse hematopoiesis (50D, 3 timepoints) | 260 s |

These values are rough references derived from the supplementary benchmark by scaling the reported per-100-step runtime by 5.

---

## 6. Summary

| Item | Value |
|------|-------|
| Core framework | [PyTorch]((https://pytorch.org/)) |
| SDE solver | [torchsde](https://github.com/google-research/torchsde) |
| OT library | [POT](https://pythonot.github.io/) |
| Example | `examples/three_gene_alpha05.ipynb` |
| GitHub repository | `scDIVIDE` |

