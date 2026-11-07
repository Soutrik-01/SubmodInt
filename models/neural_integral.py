import math
import typing
from collections.abc import Iterable
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor


def flatten_params(params: Iterable[Tensor]) -> Tensor:
    flat = [p.contiguous().view(-1) for p in params]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def compute_cc_weights(nb_steps: int) -> tuple[Tensor, Tensor]:
    lam = torch.arange(0, nb_steps + 1, 1, dtype=torch.float64).reshape(-1, 1)
    lam = torch.cos((lam @ lam.T) * math.pi / nb_steps)
    lam[:, 0] = 0.5
    lam[:, -1] = 0.5 * lam[:, -1]
    lam = lam * 2 / nb_steps
    W = torch.arange(0, nb_steps + 1, 1, dtype=torch.float64).reshape(-1, 1)
    W[torch.arange(1, nb_steps + 1, 2)] = 0
    W = 2 / (1 - W**2)
    W[0] = 1
    W[torch.arange(1, nb_steps + 1, 2)] = 0
    cc_weights = (lam.T @ W).to(torch.float32)
    steps = torch.cos(
        torch.arange(0, nb_steps + 1, 1, dtype=torch.float64).reshape(-1, 1) * math.pi / nb_steps
    ).to(torch.float32)
    return cc_weights, steps


def integrate(
    x0: Tensor,
    nb_steps: int,
    step_sizes: Tensor,
    integrand: nn.Module,
    compute_grad: bool = False,
    x_tot: Optional[Tensor] = None,
) -> Tensor:
    cc_weights, steps = compute_cc_weights(nb_steps)

    device = x0.get_device() if x0.is_cuda else "cpu"
    cc_weights, steps = cc_weights.to(device), steps.to(device)

    xT = x0 + nb_steps * step_sizes
    if not compute_grad:
        x0_t = x0.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        xT_t = xT.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        steps_t = steps.unsqueeze(0).expand(x0_t.shape[0], -1, x0_t.shape[2])
        X_steps = x0_t + (xT_t - x0_t) * (steps_t + 1) / 2
        X_steps = X_steps.contiguous().view(-1, x0_t.shape[2])
        dzs = integrand(X_steps)
        dzs = dzs.view(xT_t.shape[0], nb_steps + 1, -1)
        dzs = dzs * cc_weights.unsqueeze(0).expand(dzs.shape)
        z_est = dzs.sum(1)
        return z_est * (xT - x0) / 2
    x0_t = x0.unsqueeze(1).expand(-1, nb_steps + 1, -1)
    xT_t = xT.unsqueeze(1).expand(-1, nb_steps + 1, -1)
    x_tot = typing.cast(Tensor, x_tot) * (xT - x0) / 2
    x_tot_steps = x_tot.unsqueeze(1).expand(-1, nb_steps + 1, -1) * cc_weights.unsqueeze(
        0
    ).expand(x_tot.shape[0], -1, x_tot.shape[1])
    steps_t = steps.unsqueeze(0).expand(x0_t.shape[0], -1, x0_t.shape[2])
    X_steps = x0_t + (xT_t - x0_t) * (steps_t + 1) / 2
    X_steps = X_steps.contiguous().view(-1, x0_t.shape[2])
    x_tot_steps = x_tot_steps.contiguous().view(-1, x_tot.shape[1])

    return computeIntegrand(X_steps, integrand, x_tot_steps)


def computeIntegrand(x: Tensor, integrand: nn.Module, x_tot: Tensor) -> Tensor:
    with torch.enable_grad():
        f = integrand.forward(x)
        return flatten_params(
            torch.autograd.grad(
                f, tuple(integrand.parameters()), x_tot, create_graph=True, retain_graph=True
            )
        )


class ParallelNeuralIntegral(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, x0: Tensor, x: Tensor, integrand: nn.Module, flat_params, nb_steps: int = 20
    ) -> Tensor:
        with torch.no_grad():
            x_tot = integrate(x0, nb_steps, (x - x0) / nb_steps, integrand, False)
            ctx.integrand = integrand
            ctx.nb_steps = nb_steps
            ctx.save_for_backward(x0.clone(), x.clone())
        return x_tot

    @staticmethod
    def backward(
        ctx, *grad_outputs: Tensor
    ) -> tuple[
        Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]
    ]:
        if not next(iter(ctx.integrand.parameters())).requires_grad:
            return None, None, None, None, None
        x0, x = ctx.saved_tensors
        integrand = ctx.integrand
        nb_steps = ctx.nb_steps
        integrand_grad = integrate(
            x0, nb_steps, (x - x0) / nb_steps, integrand, True, grad_outputs[0]
        )
        x_grad = integrand(x)
        x0_grad = integrand(x0)
        return (
            -x0_grad * grad_outputs[0],
            x_grad * grad_outputs[0],
            None,
            integrand_grad,
            None,
        )
