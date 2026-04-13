## scDIVIDE

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
scDIVIDE/
└── src/scdivide/
    ├── model.py          # NeuralSDE (unified model)
    ├── networks.py       # NN definitions (Potential, Velocity, Activity)
    ├── regularizers.py   # Regularization functions + REGISTRY
    ├── sinkhorn.py       # Sinkhorn divergence (POT backend)
    └── train.py          # train_step / train_loop
```

Experiment files are in parent `TIGON/`:
- `scDIVIDE_dev/configs/` — YAML config files
- `scDIVIDE_dev/scripts/` — experiment scripts
- `scDIVIDE_dev/results/` — outputs

## NeuralSDE — Unified Model

Constructor flags control model variants:

| Flag | Values | Description |
|------|--------|-------------|
| `velocity_type` | `"potential"` / `"free"` | v = -∇φ or MLP |
| `velocity_time_dependent` | bool | v(x) or v(t,x) |
| `activity_time_dependent` | bool | β(x) or β(t,x) |
| `alpha` | float [0,1] | Death-to-birth ratio |
| `r0` | float > 0 | Baseline activity scale |

```python
from scdivide import NeuralSDE

func = NeuralSDE(
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

## Regularizers

regularizers.py で定義（`REGISTRY` 経由）:

| Key | Signature | Description |
|-----|-----------|-------------|
| `activity_smoothness` | (func, t, x) | ‖∇ₓβ‖² |
| `mass_conservation` | (func, t, x, lnw) | (E_w[g])² |
| `velocity_ratio` | (func, t, x, target=) | (‖v‖/diffusion - target)² |
| `potential_hessian` | (func, t, x) | ‖∇²φ‖²_F |
| `activity_temporal_smoothness` | (func, t1, t2, x) | ‖β(t2)-β(t1)‖²/dt² |
| `velocity_spatial_smoothness` | (func, t, x) | ‖∇ₓv‖²_F |
| `velocity_temporal_smoothness` | (func, t1, t2, x) | ‖v(t2)-v(t1)‖²/dt² |

train.py で定義（SDE 軌道に依存するため regularizers.py とは分離）:

| Key | Description |
|-----|-------------|
| `action` | WFR action: `∫ (1/N) Σ_k [0.5*(‖v_k‖² + γ‖g_k‖²) * m_k] dt` |

- m = exp(lnw), γ = `action_growth_coeff`
- 粒子数 N で割る (`torch.mean`) — 粒子数非依存
- `compute_action_on_segment(func, ts, ys, growth_coeff)` でセグメント単位に計算
- 0 でオフ（計算スキップ）、ログは `reg_losses["action"]`

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
  activity_smoothness: 0.0          # ‖∇ₓβ‖²
  activity_temporal_smoothness: 0.0 # ‖β(t2)-β(t1)‖²/dt²
  mass_conservation: 0.0            # (E_w[g])²
  velocity_ratio: 0.0               # (‖v‖/diffusion - target)²
  velocity_ratio_target: 1.0        # velocity_ratio の target 値
  potential_hessian: 0.0             # ‖∇²φ‖²_F
  velocity_spatial_smoothness: 0.0   # ‖∇ₓv‖²_F
  velocity_temporal_smoothness: 0.0  # ‖v(t2)-v(t1)‖²/dt²
  action: 0.0                       # WFR action（0 = オフ）
  action_growth_coeff: 1.0          # action 内の growth 項の重み γ
```
