"""
Training functions for NeuralSDE.

Provides:
- sample_data: sample from data with Gaussian noise
- build_optimizer / build_scheduler: configurable from YAML
- train_step: single training iteration
- train_loop: full training loop returning history
"""

import math
import time
import random

import torch
import torchsde

from .sinkhorn import compute_cost_scale, sinkhorn_divergence


# ---------------------------------------------------------------------------
# Data sampling
# ---------------------------------------------------------------------------

def sample_data(data, num_samples, noise_std, device):
    """Sample from a data tensor with additive Gaussian noise.

    Args:
        data: (N, d) tensor.
        num_samples: number of samples to draw.
        noise_std: std of isotropic Gaussian noise.
        device: torch device.

    Returns:
        (num_samples, d) tensor on *device*.
    """
    n = data.shape[0]
    dim = data.shape[1]

    if n < num_samples:
        indices = random.choices(range(n), k=num_samples)
    else:
        indices = random.sample(range(n), num_samples)

    samples = data[indices]

    if noise_std > 0:
        noise = noise_std * torch.randn(num_samples, dim, device=device, dtype=torch.float32)
        samples = samples + noise

    return samples


# ---------------------------------------------------------------------------
# Optimizer / scheduler builders
# ---------------------------------------------------------------------------

def build_optimizer(func, optimizer_config):
    """Build optimizer from config dict.

    Supported types: adam, adamw, rmsprop, sgd, radam, nadam.
    """
    opt_type = optimizer_config.get("type", "adam").lower()
    lr = optimizer_config.get("lr", 1e-3)
    wd = optimizer_config.get("weight_decay", 0.0)

    if opt_type == "rmsprop":
        return torch.optim.RMSprop(func.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == "sgd":
        momentum = optimizer_config.get("momentum", 0.0)
        return torch.optim.SGD(func.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    elif opt_type == "adamw":
        return torch.optim.AdamW(func.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == "radam":
        return torch.optim.RAdam(func.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == "nadam":
        return torch.optim.NAdam(func.parameters(), lr=lr, weight_decay=wd)
    else:
        return torch.optim.Adam(func.parameters(), lr=lr, weight_decay=wd)


def build_scheduler(optimizer, scheduler_config, niters):
    """Build LR scheduler from config dict.

    Supported types: cosine, step, exponential, plateau, warmup_cosine, none/null.
    """
    sched_type = (scheduler_config.get("type") or "cosine").lower()

    if sched_type == "cosine":
        T_max = scheduler_config.get("T_max") or niters
        eta_min = scheduler_config.get("eta_min", 1e-5)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif sched_type == "step":
        step_size = scheduler_config.get("step_size", 100)
        gamma = scheduler_config.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_type == "exponential":
        gamma = scheduler_config.get("gamma", 0.99)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif sched_type == "plateau":
        patience = scheduler_config.get("patience", 20)
        factor = scheduler_config.get("factor", 0.5)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor)
    elif sched_type == "warmup_cosine":
        warmup_iters = scheduler_config.get("warmup_iters", 50)
        T_max = scheduler_config.get("T_max") or niters
        eta_min = scheduler_config.get("eta_min", 0.0)

        def lr_lambda(step):
            if step < warmup_iters:
                return step / max(warmup_iters, 1)
            progress = (step - warmup_iters) / max(T_max - warmup_iters, 1)
            return eta_min / optimizer.defaults["lr"] + (1 - eta_min / optimizer.defaults["lr"]) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        return None


# ---------------------------------------------------------------------------
# WFR action loss
# ---------------------------------------------------------------------------

def compute_action_on_segment(func, ts, ys, growth_coeff=1.0):
    """WFR action: ∫ (1/N) Σ_k [0.5*(||v_k||² + γ||g_k||²) * m_k] dt

    粒子数 N で割る（非正規化測度を保持しつつ粒子数非依存）。
    """
    action = 0.0
    for j in range(len(ts) - 1):
        dt = ts[j + 1] - ts[j]
        drift = func.f(ts[j], ys[j])
        v = drift[:, : func.in_out_dim]
        g = drift[:, func.in_out_dim :]
        lnw = ys[j][:, func.in_out_dim :]
        m = torch.exp(torch.clamp(lnw, -20, 20)).squeeze()
        v_sq = (v ** 2).sum(dim=1)
        g_sq = (g ** 2).squeeze()
        # 粒子数 N で割る（torch.mean）
        action += (
            0.5 * torch.mean(v_sq * m)
            + 0.5 * growth_coeff * torch.mean(g_sq * m)
        ) * dt
    return action


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------

def train_step(
    func,
    num_samples,
    data_train,
    train_time,
    integral_time,
    device,
    itr,
    niters,
    # SDE
    sde_dt=0.05,
    adjoint=False,
    action_steps_per_segment=10,
    # Sinkhorn
    sinkhorn_epsilon=0.05,
    sinkhorn_epsilon_end=None,
    sinkhorn_n_iter=500,
    sinkhorn_weight=1.0,
    use_growth_weight=True,
    sinkhorn_subsample=1000,
    # Regularizers
    reg_config=None,
):
    """Run one training iteration.

    Returns:
        (total_loss, reg_losses_dict, sinkhorn_losses)

    total_loss: scalar tensor (Sinkhorn + regularizers + action).
    reg_losses_dict: {name: float} for logging.
    sinkhorn_losses: (n_segments,) detached tensor.
    """
    n_segments = len(train_time) - 1
    sinkhorn_losses_det = torch.zeros(n_segments, device=device)
    sink_loss_total = torch.tensor(0.0, device=device)

    # Cost scale from all training data
    all_data = torch.cat(data_train, dim=0)
    cost_scale = compute_cost_scale(all_data) / 2

    # Epsilon annealing
    if sinkhorn_epsilon_end is not None and niters > 1:
        progress = (itr - 1) / (niters - 1)
        current_epsilon = sinkhorn_epsilon * math.exp(
            progress * math.log(sinkhorn_epsilon_end / sinkhorn_epsilon)
        )
    else:
        current_epsilon = sinkhorn_epsilon

    # Sample from t=0
    x_curr = sample_data(data_train[0], num_samples, 0.02, device)
    initial_size = data_train[0].shape[0]
    lnw_curr = torch.log(torch.ones(num_samples, 1, device=device) / initial_size)

    sde_solver = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint

    _action_coeff = (reg_config or {}).get("action", 1.0)
    _growth_coeff = (reg_config or {}).get("action_growth_coeff", 1.0)
    action_loss = torch.tensor(0.0, device=device)

    for i in range(n_segments):
        t0 = float(integral_time[i])
        t1 = float(integral_time[i + 1])

        y0 = torch.cat([x_curr, lnw_curr], dim=1)
        ts = torch.linspace(t0, t1, action_steps_per_segment + 1, device=device)

        ys = sde_solver(func, y0, ts, method="euler", dt=sde_dt)

        # WFR action loss on this segment
        if _action_coeff > 0:
            action_loss = action_loss + compute_action_on_segment(
                func, ts, ys, growth_coeff=_growth_coeff,
            )

        y_end = ys[-1]
        z_pred = y_end[:, : func.in_out_dim]
        lnw_pred = torch.clamp(y_end[:, func.in_out_dim :], min=-20, max=20)

        # Sinkhorn divergence to target
        y_target = data_train[i + 1]
        mu = torch.exp(lnw_pred).squeeze() if use_growth_weight else None

        z_pred_sink = z_pred
        y_target_sink = y_target
        mu_sink = mu
        if sinkhorn_subsample is not None:
            n_pred = z_pred.shape[0]
            n_target = y_target.shape[0]
            if n_pred > sinkhorn_subsample:
                idx = torch.randperm(n_pred, device=device)[:sinkhorn_subsample]
                z_pred_sink = z_pred[idx]
                if mu is not None:
                    mu_sink = mu[idx]
            if n_target > sinkhorn_subsample:
                idx = torch.randperm(n_target, device=device)[:sinkhorn_subsample]
                y_target_sink = y_target[idx]

        sink_loss = sinkhorn_divergence(
            z_pred_sink, y_target_sink,
            epsilon=current_epsilon,
            n_iter=sinkhorn_n_iter,
            a=mu_sink, b=None,
            cost_scale=cost_scale,
        )

        sink_loss_total = sink_loss_total + sinkhorn_weight * sink_loss
        sinkhorn_losses_det[i] = sink_loss.detach()

        # Update state for next segment
        x_curr = z_pred
        lnw_curr = lnw_pred

    # --- WFR action regularizer ---
    reg_losses = {}
    reg_total = torch.tensor(0.0, device=device)
    if _action_coeff > 0:
        reg_losses["action"] = action_loss.item()
        reg_total = reg_total + _action_coeff * action_loss

    total_loss = sink_loss_total + reg_total
    return total_loss, reg_losses, sinkhorn_losses_det


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train_loop(
    func,
    data_train,
    train_time,
    integral_time,
    device,
    # Config dicts
    train_config,
    sde_config=None,
    sinkhorn_config=None,
    optimizer_config=None,
    scheduler_config=None,
    reg_config=None,
    output_config=None,
):
    """Full training loop.

    Args:
        func: NeuralSDE or NeuralSDE_TimeDep model (already on device).
        data_train: list of tensors per training timepoint.
        train_time: list of int indices [0, 1, ...].
        integral_time: list of float times.
        device: torch device.
        train_config: dict with niters, num_samples, gradient_clip, gradient_clip_type.
        sde_config: dict with sigma, delta, dt, adjoint.
        sinkhorn_config: dict with epsilon, epsilon_end, n_iter, weight, use_growth_weight, subsample.
        optimizer_config: dict with type, lr, weight_decay.
        scheduler_config: dict with type, T_max, eta_min, etc.
        reg_config: dict with regularizer coefficients.
        output_config: dict with verbose, print_every.

    Returns:
        dict with keys: func, loss_history, final_loss.
    """
    sde_config = sde_config or {}
    sinkhorn_config = sinkhorn_config or {}
    optimizer_config = optimizer_config or {}
    scheduler_config = scheduler_config or {}
    output_config = output_config or {}
    reg_config = reg_config or {}
    verbose = output_config.get("verbose", True)
    print_every = output_config.get("print_every", 10)

    niters = train_config["niters"]
    num_samples = train_config["num_samples"]
    gradient_clip = train_config.get("gradient_clip", None)
    gradient_clip_type = train_config.get("gradient_clip_type", "norm")

    optimizer = build_optimizer(func, optimizer_config)
    scheduler = build_scheduler(optimizer, scheduler_config, niters)

    loss_history = []
    train_start = time.time()

    if verbose:
        print(f"  Training for {niters} iterations...")

    for itr in range(1, niters + 1):
        optimizer.zero_grad()

        total_loss, reg_losses, sinkhorn_losses = train_step(
            func=func,
            num_samples=num_samples,
            data_train=data_train,
            train_time=train_time,
            integral_time=integral_time,
            device=device,
            itr=itr,
            niters=niters,
            sde_dt=sde_config.get("dt", 0.05),
            adjoint=sde_config.get("adjoint", False),
            action_steps_per_segment=sde_config.get("action_steps_per_segment", 10),
            sinkhorn_epsilon=sinkhorn_config.get("epsilon", 0.05),
            sinkhorn_epsilon_end=sinkhorn_config.get("epsilon_end"),
            sinkhorn_n_iter=sinkhorn_config.get("n_iter", 500),
            sinkhorn_weight=sinkhorn_config.get("weight", 1.0),
            use_growth_weight=sinkhorn_config.get("use_growth_weight", True),
            sinkhorn_subsample=sinkhorn_config.get("subsample", 1000),
            reg_config=reg_config,
        )

        total_loss.backward()

        if gradient_clip and gradient_clip > 0:
            if gradient_clip_type == "value":
                torch.nn.utils.clip_grad_value_(func.parameters(), gradient_clip)
            else:
                torch.nn.utils.clip_grad_norm_(func.parameters(), gradient_clip)

        optimizer.step()

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(total_loss.item())
            else:
                scheduler.step()

        loss_val = total_loss.item()
        sink_each = sinkhorn_losses.cpu().numpy().tolist()
        sink_mean = sinkhorn_losses.mean().item()

        record = {
            "itr": itr,
            "total_loss": loss_val,
            "sinkhorn_mean": sink_mean,
            "sinkhorn_each": sink_each,
        }
        record.update({f"reg_{k}": v for k, v in reg_losses.items()})
        loss_history.append(record)

        if verbose and (itr % print_every == 0 or itr == 1):
            elapsed = time.time() - train_start
            sink_str = ", ".join([f"{s:.4f}" for s in sink_each])
            reg_str = ", ".join([f"{k}={v:.4f}" for k, v in reg_losses.items()])
            extra = f", {reg_str}" if reg_str else ""
            print(
                f"    Iter {itr:4d}/{niters}: loss={loss_val:.4f}, "
                f"Sink=[{sink_str}]{extra} [{elapsed:.1f}s]"
            )

    return {
        "func": func,
        "loss_history": loss_history,
        "final_loss": loss_history[-1]["total_loss"] if loss_history else float("nan"),
        "final_sinkhorn_mean": loss_history[-1]["sinkhorn_mean"] if loss_history else float("nan"),
    }
