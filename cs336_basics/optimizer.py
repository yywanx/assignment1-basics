import torch
import math
from typing import Optional, Callable
from collections.abc import Iterable


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    grad = p.grad
                    state = self.state[p]

                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    state["step"] += 1
                    t = state["step"]

                    # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                    # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    bias_correction1 = 1 - beta1 ** t
                    bias_correction2 = 1 - beta2 ** t

                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                    # Decoupled weight decay
                    if weight_decay != 0:
                        p.add_(p, alpha=-lr * weight_decay)

                    # Adam update
                    denom = exp_avg_sq.sqrt().add_(eps)
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def get_lr_cosine_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int,
) -> float:
    if t < T_w:
        return alpha_max * t / T_w

    if t <= T_c:
        progress = (t - T_w) / (T_c - T_w)
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (
            1 + math.cos(math.pi * progress)
        )

    return alpha_min


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6

    # Convert generator to list, because model.parameters() is often an iterator.
    parameters = list(parameters)

    # Keep only params that actually have gradients.
    grads = [p.grad for p in parameters if p.grad is not None]

    if len(grads) == 0:
        return

    # Compute global L2 norm across all gradients.
    total_norm_sq = torch.zeros((), device=grads[0].device)
    for grad in grads:
        total_norm_sq += torch.sum(grad.detach() ** 2)

    total_norm = torch.sqrt(total_norm_sq)

    # Clip only if norm is larger than max_l2_norm.
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        for grad in grads:
            grad.mul_(clip_coef)