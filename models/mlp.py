import itertools
from collections.abc import Iterable, Sequence
from typing import Optional

import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        *,
        hidden_act: type[nn.Module] = nn.ReLU,
        final_act: Optional[type[nn.Module]] = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        dims = (in_dim, *tuple(hidden_dims))
        for d_in, d_out in itertools.pairwise(dims):
            layers.extend([nn.Linear(d_in, d_out), hidden_act()])
        layers.append(nn.Linear(dims[-1], out_dim))
        if final_act is not None:
            layers.append(final_act())
        self.net = nn.Sequential(*layers)
        self.reset_params()

    def reset_params(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.kaiming_uniform_(last.weight, nonlinearity="relu")
            nn.init.zeros_(last.bias)

    def layers(self) -> Iterable[nn.Linear]:
        return iter(self.net)

    def forward(self, x):
        return self.net(x)
