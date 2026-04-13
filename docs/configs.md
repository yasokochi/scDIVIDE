# scDIVIDE Config Reference

`train_loop` accepts several config dicts. Only `train_config` is required; the
rest fall back to the defaults shown below. The defaults are tuned for the
**mouse hematopoiesis** dataset — for the three-gene dataset, use the
hyperparameters reported in the manuscript.

```python
from scdivide import NeuralSDE, train_loop

func = NeuralSDE(in_out_dim=3, sigma=0.05, delta=0.1, alpha=0.5).to(device)

results = train_loop(
    func=func,
    data_train=data_train,
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

---

## `NeuralSDE` constructor

| Key | Default | Description |
|---|---|---|
| `in_out_dim` | — | Data dimensionality (required) |
| `sigma` | `0.05` | Baseline diffusion |
| `delta` | `0.1` | Offspring displacement scale |
| `alpha` | `0.0` | Death-to-birth ratio ∈ [0, 1] |
| `r0` | `1.0` | Baseline activity scale |
| `velocity_type` | `"potential"` | `"potential"` (`v = -∇φ`) or `"free"` (MLP) |
| `velocity_time_dependent` | `False` | `v(x)` vs `v(t, x)` |
| `activity_time_dependent` | `False` | `β(x)` vs `β(t, x)` |
| `velocity_hidden_dim` | `64` | Hidden width of velocity/potential MLP |
| `velocity_n_hiddens` | `4` | Hidden depth of velocity/potential MLP |
| `activity_hidden_dim` | `64` | Hidden width of activity MLP |
| `activity_n_hiddens` | `3` | Hidden depth of activity MLP |

---

## `train_config` (required)

| Key | Default | Description |
|---|---|---|
| `niters` | — | Number of training iterations (required) |
| `num_samples` | — | Particles per iteration (required) |
| `gradient_clip` | `None` | Max grad norm/value (no clipping if `None`/0) |
| `gradient_clip_type` | `"norm"` | `"norm"` or `"value"` |

---

## `sde_config`

| Key | Default | Description |
|---|---|---|
| `dt` | `0.05` | Euler-Maruyama step size |
| `adjoint` | `False` | Use `torchsde.sdeint_adjoint` |
| `action_steps_per_segment` | `10` | Sub-steps per segment for action loss |

---

## `sinkhorn_config`

| Key | Default | Description |
|---|---|---|
| `epsilon` | `0.05` | Entropic regularization |
| `epsilon_end` | `None` | If set, exponentially anneal `epsilon → epsilon_end` over training |
| `n_iter` | `500` | Sinkhorn iterations |
| `weight` | `1.0` | Weight of Sinkhorn loss in total objective |
| `use_growth_weight` | `True` | Weight predicted samples by `exp(lnw)` |
| `subsample` | `1000` | Random subsample size for Sinkhorn (predicted/target) |

---

## `optimizer_config`

| Key | Default | Description |
|---|---|---|
| `type` | `"adam"` | `adam`, `adamw`, `rmsprop`, `sgd`, `radam`, `nadam` |
| `lr` | `1e-3` | Learning rate |
| `weight_decay` | `0.0` | L2 weight decay |
| `momentum` | `0.0` | (SGD only) |

---

## `scheduler_config`

Default type is `cosine` with `T_max = niters`, `eta_min = 1e-5`.

| Key | Default | Applies to |
|---|---|---|
| `type` | `"cosine"` | `cosine`, `step`, `exponential`, `plateau`, `warmup_cosine`, `none` |
| `T_max` | `niters` | `cosine`, `warmup_cosine` |
| `eta_min` | `1e-5` | `cosine`, `warmup_cosine` |
| `step_size` | `100` | `step` |
| `gamma` | `0.1` / `0.99` | `step` / `exponential` |
| `patience` | `20` | `plateau` |
| `factor` | `0.5` | `plateau` |
| `warmup_iters` | `50` | `warmup_cosine` |

Set `type: "none"` to disable.

---

## `reg_config`

| Key | Default | Description |
|---|---|---|
| `action` | `1.0` | WFR action regularizer coefficient (`0` to disable) |
| `action_growth_coeff` | `1.0` | Weight γ of growth term inside the action |

WFR action:

```
∫ (1/N) Σ_k [ 0.5 * (‖v_k‖² + γ ‖g_k‖²) * m_k ] dt
```

where `m_k = exp(lnw_k)`.

---

## `output_config`

| Key | Default | Description |
|---|---|---|
| `verbose` | `True` | Print progress |
| `print_every` | `10` | Log interval (iterations) |
