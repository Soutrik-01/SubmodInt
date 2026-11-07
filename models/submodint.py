import math
from collections.abc import Sequence
from enum import StrEnum, auto
from typing import Optional, Self

import clap
import torch
import torch.nn as nn
import torch.nn.functional as F
from clap import arg, long
from serde import serde
from torch import Tensor

from models.mlp import MLP
from models.neural_integral import ParallelNeuralIntegral, flatten_params


class InteractionFn(StrEnum):
    A = auto()
    """Concat. Results in a (B, N, N, 2d) tensor."""
    B = auto()
    """Dot-product based interaction."""


@serde
@clap.group(title="SubmodInt config")
class SubmodIntConfig:
    """These options are used to configure SubmodInt."""
    num_layers: int = arg(long, default_value=2, value_name="N")
    """The number of layers."""
    transform_hidden_dims: list[int] = arg(long, default_value=[32], num_args="*")
    """The hidden layers of the continuous transformations."""
    interaction_hidden_dims: list[int] = arg(long, default_value=[10, 10], num_args="*")
    """The hidden layers of the interaction model."""
    modular_hidden_dims: list[int] = arg(long, default_value=[32], num_args="*")
    """The hidden layers of the modular models."""
    independent_params: bool = arg(long)
    """Use independent set of parameters for each layer."""
    fix_continuous_transforms: bool = arg(long)
    """Use fixed continuous transformations."""
    interaction_fn: InteractionFn = arg(long, default_value=InteractionFn.B)
    """Which interaction function to use."""
    no_m1: bool = arg(long)
    """Set M^(n)_1 to 0."""
    no_m2: bool = arg(long)
    """Set M^(n)_2 to 0."""
    no_phi_ccm: bool = arg(long)
    """Set phi_ccm^(n) to 1."""
    no_outer_transform: bool = arg(long)
    """Removes the outer transform."""
    nb_steps: int = arg(long, default_value=50)
    """nb_steps for ParallelNeuralIntegral."""


class Zero(nn.Module):
    def __init__(self, out_shape: Sequence[int]):
        super().__init__()
        self.shape = tuple(out_shape)

    def forward(self, x: Tensor) -> Tensor:
        return torch.zeros((x.shape[0], *self.shape), device=x.device)


def masked_sum(values: Tensor, mask: Optional[Tensor]) -> Tensor:
    """Sum (B, N, k) over set elements; with a (B, N) bool mask, padded elements are skipped."""
    if mask is not None:
        values = values * mask.unsqueeze(-1).to(values.dtype)
    return values.sum(dim=1)


class InteractionA(nn.Module):
    def __init__(self, d: int, hidden_dims: Sequence[int]):
        super().__init__()
        self.mlp = MLP(2 * d, hidden_dims, 1, final_act=nn.ReLU)

    def forward(self, _: Tensor, x1: Tensor, x2: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """_, (B, N, N, d), (B, N, N, d) -> (B, 1)"""
        g1 = self.mlp(torch.cat([x1, x2], dim=-1)).squeeze(-1)
        g2 = self.mlp(torch.cat([x2, x1], dim=-1)).squeeze(-1)
        pair = F.relu(0.5 * (g1 + g2))
        if mask is not None:
            pair = pair * (mask.unsqueeze(2) & mask.unsqueeze(1)).to(pair.dtype)
        return pair.sum(dim=(1, 2)).unsqueeze(-1)


class InteractionB(nn.Module):
    """I(S) = Sum(x, y in S) a(x, y), where a(x, y) = <MLP(x), MLP(y)>."""
    def __init__(self, d: int, hidden_dims: Sequence[int]):
        super().__init__()
        assert len(hidden_dims) > 1
        self.mlp = MLP(d, hidden_dims[:-1], hidden_dims[-1], final_act=nn.ReLU)

    def forward(self, x: Tensor, *_: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """(B, N, d) -> (B, 1)"""
        return (masked_sum(self.mlp(x), mask)**2).sum(dim=1, keepdim=True)


SCALE_INIT = math.log(math.e - 1)


class Phi1Fixed(nn.Module):
    """phi^(1)(x) = s (1 - e^(-C x)), with a learned scale s > 0 (as the neural phi^(1) has alpha)
    satisfies
    - phi(0) = 0,
    - phi(x) >= 0,
    - phi'(x) >= 0, and
    - ...
    for all x >= 0.
    """

    def __init__(self, C: float):
        super().__init__()
        self.beta = torch.tensor(C, dtype=torch.float32)
        self.a = nn.Parameter(torch.tensor(SCALE_INIT, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        return (F.softplus(self.a) + 1e-6) * (1.0 - torch.exp(-self.beta * x))


class Phi1(nn.Module):
    """phi_1(x) = Integral(t, x_0, x) e^(-Ct) h(t) dt, where h(t) > 0."""
    class Integrand(nn.Module):
        """phi'(x) =  e^(-Ct)h(t), where h(t) = alpha * softplus(-ReLU(NN(x)))."""
        def __init__(self, C: float, hidden_dims: Sequence[int]):
            super().__init__()
            self.C = torch.tensor(C)
            self.a = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.mlp = MLP(1, hidden_dims, 1, final_act=nn.ReLU)

        def forward(self, x: Tensor) -> Tensor:
            alpha = F.softplus(self.a) + 1e-6
            return alpha * torch.exp(-self.C * x) * F.softplus(-self.mlp(x))

    def __init__(self, C: float, hidden_dims: Sequence[int], nb_steps: int):
        super().__init__()
        self.integrand = self.Integrand(C, hidden_dims)
        self.x0 = nn.Parameter(torch.tensor([[0.0]], dtype=torch.float32))
        self.nb_steps = nb_steps

    def forward(self, x: Tensor) -> Tensor:
        """(B, 1) -> (B, 1)"""
        return ParallelNeuralIntegral.apply(
            self.x0,
            x,
            self.integrand,
            flatten_params(self.integrand.parameters()),
            self.nb_steps,
        )


class IncreasingConcaveTransformationFixed(nn.Module):
    """f(x) = s log(1 + x), with a learned scale s > 0 (as the neural transformation has alpha)"""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(SCALE_INIT, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        return (F.softplus(self.a) + 1e-6) * torch.log1p(x)


class IncreasingConcaveTransformation(nn.Module):
    """psi(x) = Integral(t, 0, x) h(t) dt, where h(t) > 0.
    satisfies
    - psi(0) = 0,
    - psi(x) >= 0, and
    - psi'(x) >= 0
    for all x >= 0. psi''(x) <= 0 is not enforced.
    """

    class Integrand(nn.Module):
        def __init__(self, hidden_dims: Sequence[int]):
            super().__init__()
            self.mlp = MLP(1, hidden_dims, 1, final_act=nn.ReLU)
            self.alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        def forward(self, x: Tensor) -> Tensor:
            """(B, 1) -> (B, 1)"""
            return (F.softplus(self.alpha) + 1e-6) * F.softplus(-self.mlp(x))

    def __init__(self, hidden_dims: Sequence[int], nb_steps: int):
        super().__init__()
        self.x0 = nn.Parameter(torch.tensor([[0.0]], dtype=torch.float32))
        self.integrand = self.Integrand(hidden_dims)
        self.nb_steps = nb_steps

    def forward(self, x: Tensor) -> Tensor:
        """(B, 1) -> (B, 1)"""
        return ParallelNeuralIntegral.apply(
            self.x0,
            x,
            self.integrand,
            flatten_params(self.integrand.parameters()),
            self.nb_steps,
        )


class SubmodInt(nn.Module):
    """F(S) = F^(N) (S),
    where
        F^(1) (S) = phi_inter^(1) (I^(1)(S) + M^(1)_1 (S))
                    + phi_ccm^(1) (M^(1)_2 (S)),
    and for 1 < n <= N,
        F^(n) (S) = phi_outer^(n) (
            alpha * (phi_inter^(n) (I^(n) (S) + M^(n)_1 (S))
                      + phi_ccm^(n) (M^(n)_2 (S)))
            + (1 - alpha) * F^(n - 1) (S)
        ),
    where alpha in (0, 1). For each n such that 1 <= n <= N:
    - M^(n)_1 and M^(n)_2 are monotone increasing positive set functions,
    - I^(n) (S) = Sum((i, j) in S) a(z_i, z_j) for all S subseteq V, where
      a : Z x Z -> R+ is such that a(x, y) = a(y, x) for all x, y in Z.
    - phi_inter^(n) : R+ -> R+ satisfies
      - phi_inter^(n) (0) = 0,
      - phi_inter^(n) (x) >= 0,
      - phi_inter^(n)' (x) >= 0, and
      - ...
      for all x >= 0.
    - phi_ccm^(n) : R+ -> R+ satisfies
      - phi_ccm^(n) (0) = 0,
      - phi_ccm^(n) (x) >= 0,
      - phi_ccm^(n)' (x) >= 0, and
      - phi_ccm^(n)'' (x) <= 0,
      for all x >= 0.
    For each n such that 1 < n <= N, phi_outer^(n) : R+ -> R+ satisfies
    - phi_outer^(n) (0) = 0,
    - phi_outer^(n) (x) >= 0,
    - phi_outer^(n)' (x) >= 0, and
    - phi_outer^(n)'' (x) <= 0,
    for all x >= 0.
    """

    def __init__(
        self,
        interaction_models: nn.ModuleList,
        m1s: nn.ModuleList,
        m2s: nn.ModuleList,
        phi_inters: nn.ModuleList,
        phi_ccms: nn.ModuleList,
        phi_outers: nn.ModuleList,
    ):
        super().__init__()

        self.num_layers = len(interaction_models)
        assert all(len(m) == self.num_layers for m in (m1s, m2s, phi_inters, phi_ccms, phi_outers))
        self.interaction_models = interaction_models
        self.m1s = m1s
        self.m2s = m2s
        self.phi_inters = phi_inters
        self.phi_ccms = phi_ccms
        self.phi_outers = phi_outers
        self.alpha = nn.Parameter(torch.rand(1, dtype=torch.float32))

    @classmethod
    def from_config(
        cls,
        *,
        feature_dim: int,
        gamma: float,
        max_set_size: float,
        config: SubmodIntConfig,
    ) -> Self:
        C = (gamma / 2.0) * math.log(gamma * (max_set_size - 1) + 1)

        interaction_models: nn.ModuleList
        interaction_cls: type[nn.Module]
        match config.interaction_fn:
            case InteractionFn.A:
                interaction_cls = InteractionA
            case InteractionFn.B:
                interaction_cls = InteractionB
        if config.independent_params:
            interaction_models = nn.ModuleList(
                interaction_cls(d=feature_dim, hidden_dims=config.interaction_hidden_dims)
                for _ in range(config.num_layers)
            )
        else:
            interaction_model = interaction_cls(
                d=feature_dim, hidden_dims=config.interaction_hidden_dims
            )
            interaction_models = nn.ModuleList(interaction_model for _ in range(config.num_layers))

        m1s: nn.ModuleList
        m2s: nn.ModuleList
        if config.independent_params:
            m1s = nn.ModuleList(
                MLP(feature_dim, config.modular_hidden_dims, 1, final_act=nn.ReLU)
                for _ in range(config.num_layers)
            )
            m2s = nn.ModuleList(
                MLP(feature_dim, config.modular_hidden_dims, 1, final_act=nn.ReLU)
                for _ in range(config.num_layers)
            )
        else:
            m1 = MLP(feature_dim, config.modular_hidden_dims, 1, final_act=nn.ReLU)
            m2 = MLP(feature_dim, config.modular_hidden_dims, 1, final_act=nn.ReLU)
            m1s = nn.ModuleList(m1 for _ in range(config.num_layers))
            m2s = nn.ModuleList(m2 for _ in range(config.num_layers))
        zero_model = Zero((1, 1))
        if config.no_m1:
            m1s = nn.ModuleList(zero_model for _ in range(config.num_layers))
        if config.no_m2:
            m2s = nn.ModuleList(zero_model for _ in range(config.num_layers))

        phi_inters: nn.ModuleList
        phi_ccms: nn.ModuleList
        phi_outers: nn.ModuleList
        if config.fix_continuous_transforms:
            phi_inters = nn.ModuleList(Phi1Fixed(C) for _ in range(config.num_layers))
            phi_ccms = nn.ModuleList(
                IncreasingConcaveTransformationFixed() for _ in range(config.num_layers)
            )
            if config.no_outer_transform:
                phi_outers = nn.ModuleList(nn.Identity() for _ in range(config.num_layers))
            else:
                phi_outers = nn.ModuleList((
                    nn.Identity(),
                    *(IncreasingConcaveTransformationFixed() for _ in range(config.num_layers - 1)),
                ))
        else:
            phi_inters = nn.ModuleList(
                Phi1(C, config.transform_hidden_dims, config.nb_steps)
                for _ in range(config.num_layers)
            )
            phi_ccms = nn.ModuleList(
                IncreasingConcaveTransformation(config.transform_hidden_dims, config.nb_steps)
                for _ in range(config.num_layers)
            )
            if config.no_outer_transform:
                phi_outers = nn.ModuleList(nn.Identity() for _ in range(config.num_layers))
            else:
                phi_outers = nn.ModuleList((
                    nn.Identity(),
                    *(
                        IncreasingConcaveTransformation(
                            config.transform_hidden_dims, config.nb_steps
                        )
                        for _ in range(config.num_layers - 1)
                    ),
                ))

        if config.no_m2:
            zero_model = Zero((1,))
            phi_ccms = nn.ModuleList(zero_model for _ in range(config.num_layers))

        if config.no_phi_ccm:
            phi_ccms = nn.ModuleList(nn.Identity() for _ in range(config.num_layers))


        return cls(
            interaction_models=interaction_models,
            m1s=m1s,
            m2s=m2s,
            phi_inters=phi_inters,
            phi_ccms=phi_ccms,
            phi_outers=phi_outers,
        )

    def g(
        self, layer: int, x: Tensor, x1: Tensor, x2: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        if mask is None:
            return self.phi_inters[layer](
                self.interaction_models[layer](x, x2, x1) + self.m1s[layer](x).sum(dim=1)
            ) + self.phi_ccms[layer](self.m2s[layer](x).sum(dim=1))
        return self.phi_inters[layer](
            self.interaction_models[layer](x, x2, x1, mask=mask)
            + masked_sum(self.m1s[layer](x), mask)
        ) + self.phi_ccms[layer](masked_sum(self.m2s[layer](x), mask))

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        x: (B, N, d)
        mask: (B, N), optional; False marks padding
        output: (B, 1)
        """
        B, N, d = x.shape
        x1 = x.unsqueeze(1).expand(B, N, N, d)
        x2 = x.unsqueeze(2).expand(B, N, N, d)
        y = self.g(0, x, x1, x2, mask)
        for layer in range(1, self.num_layers):
            alpha = F.sigmoid(self.alpha)
            y = self.phi_outers[layer](
                alpha * self.g(layer, x, x1, x2, mask) + (1 - alpha) * y
            )
        return y
