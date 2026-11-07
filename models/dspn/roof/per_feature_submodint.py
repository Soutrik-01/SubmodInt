"""SubmodInt as the set function of DSPN, for the ImageNet-100 experiment (dspn/).

DSPN scores a set of images in two steps: a feature network maps each image to 2048 non-negative features, and a
set function maps s, the sum of these feature vectors over the set, to the score. The DSPN authors' set function
(models/dspn/roof/dsf.py) applies four fixed concave functions (log(1 + x), 1 - exp(-x), sqrt(x) and x) to each s_k
and adds the results up with learned weights >= 0. PerFeatureSubmodInt replaces it by a SubmodInt on each feature k,
applied to s_k, and adds up their outputs with learned weights c_k >= 0.

SubmodInt's inputs are sums over the elements of the set: the pairwise interaction I = sum over pairs (u, v) of
<g(x_u), g(x_v)>, and the modular terms M1 and M2. With a learned weight per feature in place of the networks g, M1
and M2 (q_k, b_k and a_k, all >= 0), they depend on the set only through s_k: I_k = (q_k s_k)^2, M1_k = b_k s_k and
M2_k = a_k s_k. SubmodInt's two layers of learned increasing transforms are then applied to these, with the same
transforms for every feature.

These transforms are integrals of small neural networks, which SubmodInt computes by numerical integration at every
input; for batch x 2048 inputs, several times per training step, that is too slow. TabulatedTransform therefore
computes each transform once per optimizer step at 4096 points and interpolates linearly between them. SubmodInt_F
(fixed=True) has closed-form transforms instead, which are evaluated directly.
"""

import math

import torch
import torch.nn as nn

from models.submodint import InteractionFn, SubmodInt, SubmodIntConfig


def _grid_cell(x, log_step, num_points):
    """For the points expm1(i * log_step), i < num_points: the index i of the interval [point i, point i + 1]
    holding x (the last one beyond the last point), x's position in it (0 to 1) and its width."""
    i = (torch.log1p(x.clamp(min=0)) / log_step).floor().clamp(max=num_points - 2)
    left, right = torch.expm1(i * log_step), torch.expm1((i + 1) * log_step)
    return i.long(), (x - left) / (right - left), right - left


@torch.compile(dynamic=True)
def _interpolate(x, table, log_step: float, num_points: int):
    i, t, _ = _grid_cell(x, log_step, num_points)
    left, right = table[i], table[i + 1]
    return left + t * (right - left)


@torch.compile(dynamic=True)
def _interpolate_backward(x, grad, table, log_step: float, num_points: int, copies: int):
    """The gradient for x, and what the table's gradient is built from: for each input, an index into `copies`
    stacked copies of the table and the gradient for the left and the right point of its interval; inputs at
    exactly 0 are left out of these and returned as one sum, their gradient for point 0."""
    i, t, width = _grid_cell(x, log_step, num_points)
    left, right = table[i], table[i + 1]
    grad_right = grad * t
    zero = x == 0
    position = torch.arange(x.numel(), device=x.device).view_as(x)
    index = torch.where(zero, position % (num_points - 1), i) + (position % copies) * num_points
    return (grad * (right - left) / width, index, torch.where(zero, 0.0, grad - grad_right),
            torch.where(zero, 0.0, grad_right), (grad * zero).sum())


class LogGridInterpolation(torch.autograd.Function):
    """Linear interpolation, at x, of a table of values at the points expm1(i * log_step), i < num_points, with
    gradients for x and for the table. The table's gradient is added up with index_add_, into `copies` copies of
    the table that are summed at the end; torch.bincount would make the CPU wait for the GPU on every call."""

    copies = 64

    @staticmethod
    def forward(ctx, x, table, log_step, num_points):
        ctx.save_for_backward(x, table)
        ctx.log_step, ctx.num_points = log_step, num_points
        return _interpolate(x, table, log_step, num_points)

    @staticmethod
    def backward(ctx, grad):
        x, table = ctx.saved_tensors
        copies, n = LogGridInterpolation.copies, ctx.num_points
        grad_x, index, grad_left, grad_right, grad_at_zero = _interpolate_backward(
            x, grad.contiguous(), table, ctx.log_step, n, copies)
        index = index.reshape(-1)
        grad_table = table.new_zeros(copies * n).index_add_(0, index, grad_left.reshape(-1).to(table.dtype))
        grad_table = grad_table.index_add_(0, index + 1, grad_right.reshape(-1).to(table.dtype))
        grad_table = grad_table.view(copies, n).sum(0)
        grad_table[0] += grad_at_zero.to(table.dtype)
        return grad_x, grad_table, None, None


class TabulatedTransform(nn.Module):
    """A learned transform of SubmodInt, phi(x) = the integral of phi.integrand from phi.x0 to x, read from a table.

    The table holds phi at 4096 points from 0 to 1e6, evenly spaced in log(1 + x), each computed from the previous
    one by Simpson's rule. phi is interpolated linearly between the points (and extrapolated beyond 1e6), which keeps
    it increasing, and concave where phi is. The table is computed on first use and reused until project() of
    PerFeatureSubmodInt resets it after the optimizer step.
    """

    def __init__(self, phi, x_max=1e6, num_points=4096):
        super().__init__()
        self.phi = phi
        self.num_points = num_points
        self.log_step = math.log1p(x_max) / (num_points - 1)
        self.table = None

    def compute_table(self, device):
        n = self.num_points
        points = torch.expm1(torch.arange(n, device=device, dtype=self.phi.x0.dtype) * self.log_step)
        x0 = self.phi.x0.view(())
        below_x0 = torch.linspace(0, 1, 129, device=device) * x0
        f = self.phi.integrand(torch.cat([points, (points[:-1] + points[1:]) / 2, below_x0]).unsqueeze(1)).squeeze(1)
        f_points, f_midpoints, f_below_x0 = f[:n], f[n:2 * n - 1], f[2 * n - 1:]
        intervals = (points[1:] - points[:-1]) / 6 * (f_points[:-1] + 4 * f_midpoints + f_points[1:])
        simpson_weights = torch.ones(129, device=device)
        simpson_weights[1:-1:2], simpson_weights[2:-1:2] = 4, 2
        zero_to_x0 = (simpson_weights * f_below_x0).sum() * x0 / 384
        return torch.cat([intervals.new_zeros(1), intervals.cumsum(0)]) - zero_to_x0

    def forward(self, x):
        if self.table is None or (torch.is_grad_enabled() and not self.table.requires_grad):
            self.table = self.compute_table(x.device)
        return LogGridInterpolation.apply(x.contiguous(), self.table, self.log_step, self.num_points)


def _tabulate(transform):
    return TabulatedTransform(transform) if hasattr(transform, "integrand") else transform


class PerFeatureSubmodInt(nn.Module):
    """f(s) = sum_k c_k y_k for s of shape (batch, dim), y_k being SubmodInt with two layers on feature k (see the
    module docstring); gamma and max_set_size set its interaction transform as in SubmodInt.from_config. DSPN
    calls project() after every optimizer step: it keeps a, b, q and c >= 0 and resets the transforms' tables."""

    def __init__(self, dim, fixed, max_set_size, gamma=2.5):
        super().__init__()
        config = SubmodIntConfig(
            num_layers=2, transform_hidden_dims=[32], interaction_hidden_dims=[10, 10], modular_hidden_dims=[32],
            independent_params=False, fix_continuous_transforms=fixed, interaction_fn=InteractionFn.B,
            no_m1=False, no_m2=False, no_phi_ccm=False, no_outer_transform=False, nb_steps=50,
        )
        net = SubmodInt.from_config(feature_dim=1, gamma=gamma, max_set_size=max_set_size, config=config)
        self.phi_inters = nn.ModuleList(_tabulate(m) for m in net.phi_inters)
        self.phi_ccms = nn.ModuleList(_tabulate(m) for m in net.phi_ccms)
        self.phi_outers = nn.ModuleList(_tabulate(m) for m in net.phi_outers)
        self.alpha = net.alpha
        self.a = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.full((dim,), 0.1))
        self.q = nn.Parameter(torch.full((dim,), 0.1))
        self.c = nn.Parameter(torch.rand(dim) / math.sqrt(dim))

    def forward(self, s):
        s = s.clone()
        i_plus_m1 = (self.q * s) ** 2 + self.b * s
        m2 = self.a * s
        f1 = self.phi_inters[0](i_plus_m1) + self.phi_ccms[0](m2)
        alpha = torch.sigmoid(self.alpha)
        f2 = self.phi_outers[1](alpha * (self.phi_inters[1](i_plus_m1) + self.phi_ccms[1](m2)) + (1 - alpha) * f1)
        return (f2 * self.c).sum(-1, keepdim=True)

    def project(self):
        for p in (self.a, self.b, self.q, self.c):
            p.data.clamp_(min=0)
        for m in self.modules():
            if isinstance(m, TabulatedTransform):
                m.table = None
