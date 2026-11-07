from typing import Union, List
from numbers import Number
import torch.nn as nn
import torch
from .layers import PosLinear, MiLU, MinComponent
import logging

class Modular(nn.Module):
    def __init__(self, items, max_val: int):
        super(Modular, self).__init__()
        k = len(items)
        self.w = nn.Parameter(torch.randint(0, max_val, (k, 1)).float())
        logging.info(f"{self.w.squeeze()}")

    def forward(self, x):
        return torch.matmul(x, self.w)


class SCMM(nn.Module):
    def __init__(self, in_dim, out_dim, alpha: float = 1.0, set_activation=True):
        super(SCMM, self).__init__()
        self.set_activation = set_activation
        self.linear = PosLinear(in_dim, out_dim)
        self.activation = MiLU(alpha=alpha)

    def forward(self, x):
        y_pred = None
        if self.set_activation:
            x = self.linear(x)
            y_pred = self.activation(x)
        else:
            y_pred = self.linear(x)
        return y_pred

    def relu(self):
        self.linear.relu()


class DSF(nn.Module):
    def __init__(self, in_dim, out_dim, max_out, hidden_sizes: List[int], alpha: Union[List[float], float] = 1.0):
        super(DSF, self).__init__()
        if isinstance(alpha, Number):
            alpha = [alpha] * len(hidden_sizes)

        self.layers = nn.ParameterList()
        self.layers.append(SCMM(in_dim, hidden_sizes[0], alpha=alpha[0]))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(SCMM(hidden_sizes[i-1], hidden_sizes[i], alpha=alpha[i]))
        self.layers.append(SCMM(hidden_sizes[-1], out_dim, alpha=max_out, set_activation=False))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def relu(self):
        for layer in self.layers:
            layer.relu()


class ExtendedDSF(nn.Module):
    def __init__(self, in_dim, hidden_sizes: List[int], alpha: Union[List[float], float] = 1.0):
        super(ExtendedDSF, self).__init__()
        if isinstance(alpha, Number):
            alpha = [alpha] * len(hidden_sizes)

        self.layers = nn.ParameterList()
        self.layers.append(SCMM(in_dim, hidden_sizes[0], alpha=alpha[0]))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(SCMM(hidden_sizes[i-1], hidden_sizes[i], alpha=alpha[i]))
        self.layers.append(MinComponent())

    def forward(self, x):
        x = self.layers[0].linear(x).sum(dim=1)
        x = self.layers[0].activation(x)
        for layer in self.layers[1:]:
            x = layer(x)
        return x

    def relu(self):
        for layer in self.layers:
            layer.relu()


class DSFWrapper(nn.Module):
    def __init__(self, m, n, dsfs: List[DSF]):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros((m, n))).float()
        assert n == len(dsfs)
        self.dsfs = dsfs

    def forward(self, x):
        mask = (x > 0).float()
        masked_weights = mask * self.weights
        s = torch.tensor([0.0]).cuda()
        for b, dsf in enumerate(self.dsfs):
            s += dsf(masked_weights[:, :, b]).mean()
        return s

    def project_weights(self, z=1):
        """v array of shape (n_features, n_samples)."""
        v = self.weights.T
        p, n = v.shape
        u = torch.sort(v, dim=0, descending=True)[0]
        pi = torch.cumsum(u, dim=0) - z
        ind = torch.reshape(torch.arange(p) + 1, (-1, 1)).cuda()
        mask = (u - pi / ind) > 0
        rho = p - 1 - torch.argmax(mask.flip([0]).int(), dim=0)
        theta = pi[tuple([rho, torch.arange(n).cuda()])] / (rho + 1)
        w = torch.maximum(v - theta, torch.tensor(0).cuda())
        self.weights.data = w.T
