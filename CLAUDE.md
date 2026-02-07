## scBURST

Growth-dependent diffusion SDE for unbalanced optimal transport.

### PDE

```
∂ρ/∂t = -∇·(v ρ) + (σ² + b δ²)/2 Δρ + g ρ
```

### (β, α) Parametrization

| Symbol | Role | Type |
|--------|------|------|
| β(x) ∈ ℝ | Log-activity (NN output, unconstrained) | Estimated |
| α ∈ [0,1] | Death-to-birth ratio | Given (prior) |
| r₀ > 0 | Baseline activity scale | Fixed |
| δ² | Offspring displacement variance | Estimated |
| σ² | Non-division noise (regularization) | Fixed (small) |

**Derived quantities:**
- Birth rate: `b(x) = r₀ exp(β(x))`
- Death rate: `d(x) = r₀ exp(β(x)) · α`
- Net growth: `g(x) = r₀ exp(β(x)) · (1 - α)`
- Effective diffusion: `D_eff(x) = (σ² + b(x) δ²) / 2`

### SDE (Euler-Maruyama)

```
dz = v(z) dt + √(σ² + b·δ²) dW
d(lnw) = g dt
```

- v(z) = -∇φ(z) (potential) or MLP (free-form)
- Diffusion depends on **birth rate b**, not net growth g

### Default: α=0, r₀=1

When α=0 and r₀=1: `b = exp(β)`, `g = exp(β)` = b, so diffusion = `√(σ² + g·δ²)`.
This recovers non-negative growth with exp instead of softplus.

## Directory Structure

```
scBURST/
└── src/scburst/
    ├── model.py          # BranchingSDE (unified model)
    ├── networks.py       # NN definitions (Potential, Velocity, Activity)
    ├── regularizers.py   # Regularization functions + REGISTRY
    ├── sinkhorn.py       # Sinkhorn divergence (POT backend)
    └── train.py          # train_step / train_loop
```

Experiment files are in parent `TIGON/`:
- `scBURST_dev/configs/` — YAML config files
- `scBURST_dev/scripts/` — experiment scripts
- `scBURST_dev/results/` — outputs

## BranchingSDE — Unified Model

Constructor flags control model variants:

| Flag | Values | Description |
|------|--------|-------------|
| `velocity_type` | `"potential"` / `"free"` | v = -∇φ or MLP |
| `velocity_time_dependent` | bool | v(x) or v(t,x) |
| `activity_time_dependent` | bool | β(x) or β(t,x) |
| `alpha` | float [0,1] | Death-to-birth ratio |
| `r0` | float > 0 | Baseline activity scale |

```python
from scburst import BranchingSDE

func = BranchingSDE(
    in_out_dim=2, sigma=0.05, delta=0.1,
    velocity_type="potential",
    velocity_time_dependent=False,
    activity_time_dependent=True,
    alpha=0.0, r0=1.0,
    velocity_hidden_dim=32, velocity_n_hiddens=4,
    activity_hidden_dim=32, activity_n_hiddens=3,
)

# Public API — always takes (t, z)
beta = func.activity(t, z)      # (batch, 1), unconstrained
b = func.birth_rate(t, z)       # (batch, 1), r₀ exp(β) >= 0
d = func.death_rate(t, z)       # (batch, 1), r₀ exp(β) α >= 0
g = func.growth(t, z)           # (batch, 1), r₀ exp(β)(1-α)
v = func.velocity(t, z)         # (batch, d)
phi = func.potential(t, z)      # velocity_type="potential" only

# torchsde interface
drift = func.f(t, y)    # y = cat(z, lnw); drift = [v, g]
diff  = func.g(t, y)    # diffusion = sqrt(σ² + b·δ²)
```

## Networks (networks.py)

| Class | Input | Output | Usage |
|-------|-------|--------|-------|
| `PotentialNetwork` | x | scalar | φ(x), `.gradient(x)` |
| `TimeDependentPotentialNetwork` | t, x | scalar | φ(t,x), `.gradient(t, x)` |
| `VelocityNetwork` | x | d-dim | v(x) free-form |
| `TimeDependentVelocityNetwork` | t, x | d-dim | v(t,x) free-form |
| `ActivityNetwork` | x | scalar (unconstrained) | β(x) |
| `TimeDependentActivityNetwork` | t, x | scalar (unconstrained) | β(t,x) |

## Regularizers (regularizers.py)

| Key | Signature | Description |
|-----|-----------|-------------|
| `activity_smoothness` | (func, t, x) | ‖∇ₓβ‖² |
| `mass_conservation` | (func, t, x, lnw) | (E_w[g])² |
| `velocity_ratio` | (func, t, x, target=) | (‖v‖/diffusion - target)² |
| `potential_hessian` | (func, t, x) | ‖∇²φ‖²_F |
| `activity_temporal_smoothness` | (func, t1, t2, x) | ‖β(t2)-β(t1)‖²/dt² |
| `velocity_spatial_smoothness` | (func, t, x) | ‖∇ₓv‖²_F |
| `velocity_temporal_smoothness` | (func, t1, t2, x) | ‖v(t2)-v(t1)‖²/dt² |

## Config YAML

```yaml
model:
  velocity_type: "potential"
  velocity_time_dependent: false
  activity_time_dependent: true
  alpha: 0.0
  r0: 1.0
  velocity:
    hidden_dim: 32
    n_hiddens: 4
    activation: "tanh"
    arch: "mlp"
  activity:
    hidden_dim: 32
    n_hiddens: 3

regularizers:
  activity_smoothness: 0.0
  activity_temporal_smoothness: 0.0
  mass_conservation: 0.0
  velocity_ratio: 0.0
```
