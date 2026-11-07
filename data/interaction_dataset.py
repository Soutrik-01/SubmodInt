import itertools
import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Beta, Distribution, Gamma
from torch.utils.data import Dataset

from models.mlp import MLP


class PhiInter:
    def __init__(self, C: float, a: Tensor, b: Tensor):
        self.C = C
        self.A = 2 / C**2 - 2 * a / C
        self.B = (b + a**2) / C - 2 * a / C**2 + 2 / C**3

    def __call__(self, x: Tensor) -> Tensor:
        return torch.exp(-self.C * x) * (-(x**2) / self.C - self.A * x - self.B) + self.B


class PhiCCM:
    def __init__(self, a: Tensor, b: Tensor):
        self.a = a
        self.b = b

    def __call__(self, x: Tensor) -> Tensor:
        return self.a * torch.log(1 + self.b * x)


class PhiOuter:
    def __init__(self, a: Tensor, pow: Tensor):
        self.a = a
        self.pow = pow

    def __call__(self, x: Tensor) -> Tensor:
        return self.a * x**self.pow


class Interaction:
    def __init__(self, d: int, interaction_arity: int, dist: Distribution):
        self.mlp = MLP(d, (256, 256), 128)
        for layer in self.mlp.layers():
            if isinstance(layer, nn.Linear):
                with torch.no_grad():
                    layer.weight.data.abs_()
                    layer.bias.copy_(dist.sample(layer.bias.shape))

        match interaction_arity:
            case 2: self.interaction = self.interaction_2
            case 3: self.interaction = self.interaction_3
            case 4: self.interaction = self.interaction_4
            case _: raise NotImplementedError

    def interaction_2(self, x: Tensor) -> Tensor:
        return torch.sum(x.sum(dim=1)**2, dim=1)

    def interaction_3(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def interaction_4(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, x: Tensor, mask: Tensor) -> Tensor:
        return self.interaction(self.mlp(x * mask.unsqueeze(-1)))


class InteractionDataset(Dataset):
    def __init__(
        self,
        N: int,
        k: int,
        d: int,
        gamma: float,
        interaction_arity: int,
        num_layers: int,
    ):
        self.sets = torch.rand(N, k, d)
        self.masks = torch.arange(k) < torch.randint(k + 1, (N, 1))
        self.sets *= self.masks.unsqueeze(-1)
        self.ys = self.compute_ys(gamma, num_layers, interaction_arity)

    def compute_ys(self, gamma: float, num_layers: int, interaction_arity: int) -> Tensor:
        """F(S) = F^(N) (S).
            F^(1)(S) = G^(1)(S)
            F^(n)(S) = alpha^(n) phi^(n)_outer(G^(n)(S) + (1 - alpha) F^(n - 1)(S)),
        where 1 < n <= N = num_layers, and
            G^(n)(S) = phi^(1)_inter(I(S) + M_1(S)) + phi^(1)_ccm(M_2(S)).
        """
        _, k, d = self.sets.shape

        dist_1 = Beta(5, 5)
        alphas = [dist_1.sample((1,)) for _ in range(num_layers - 1)]

        dist_2 = Gamma(1, 2)
        interaction_fns = [Interaction(d, interaction_arity, dist_2) for _ in range(num_layers)]
        C = (gamma / 2.0) * math.log(gamma * (k - 1) + 1.0)
        phi_inters = [
            PhiInter(C, 1 + dist_2.sample((1,)), dist_2.sample((1,))) for _ in range(num_layers)
        ]

        m1s = [MLP(d, (256, 256), 1) for _ in range(num_layers)]
        m2s = [MLP(d, (256, 256), 1) for _ in range(num_layers)]
        for m in itertools.chain(m1s, m2s):
            for layer in m.layers():
                if isinstance(layer, nn.Linear):
                    with torch.no_grad():
                        layer.weight.data.abs_()
                        layer.bias.copy_(dist_2.sample(layer.bias.shape).requires_grad_())

        phi_ccms = [PhiCCM(dist_2.sample((1,)), dist_2.sample((1,))) for _ in range(num_layers)]
        phi_outers = [
            PhiOuter(dist_2.sample((1,)), dist_1.sample((1,))) for _ in range(num_layers - 1)
        ]

        def g(x: Tensor, mask: Tensor, layer: int) -> Tensor:
            return phi_inters[layer](
                1 / gamma + interaction_fns[layer](x, mask)
            ) + phi_ccms[layer]((m2s[layer](x) * mask.unsqueeze(-1)).sum(dim=1).squeeze(-1))

        ys = g(self.sets, self.masks, 0)
        for layer in range(1, num_layers):
            alpha = alphas[layer - 1]
            ys = phi_outers[layer - 1](alpha * g(self.sets, self.masks, layer) + (1 - alpha) * ys)

        return ys.detach()

    def __len__(self):
        return self.sets.shape[0]

    def __getitem__(self, index):
        return self.sets[index], self.ys[index]
