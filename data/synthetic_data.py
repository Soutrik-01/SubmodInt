import math
from collections.abc import Callable
from enum import StrEnum, auto
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import random_split

class SynPhi1(StrEnum):
    One = auto()
    Two = auto()


class SynPhi2(StrEnum):
    One = auto()
    Two = auto()


class SynPsi(StrEnum):
    One = auto()
    Two = auto()


class FixedFunctions:
    """
    Data generator functions:
      - a(x,y) in [1/gamma, 1], symmetric
      - m1(z) in [0,1]
      - m2(z) >= 0
      - phi(t) increasing with phi'' <= -C phi', we choose phi(t) = 1 - exp(-C t)
      - psi(u) concave, we choose psi(u) = log(1 + u)
    """

    def __init__(
        self,
        d: int,
        k: int,
        phi1: Optional[SynPhi1],
        phi2: Optional[SynPhi2],
        psi: Optional[SynPsi],
        gamma: float,
        num_layers: int,
        alpha: float = 0.5,
    ):
        self.d = d
        self.k = k
        self.gamma = float(gamma)
        self.minv = 1.0 / self.gamma
        self.range = 1.0 - self.minv
        self.alpha = alpha

        self.w_a = torch.randn(d, dtype=torch.float32)
        self.w_m1 = torch.randn(d, dtype=torch.float32)
        self.w_m2 = torch.randn(d, dtype=torch.float32)
        self.b_m2 = torch.randn((), dtype=torch.float32) * 0.0

        self.C = (self.gamma / 2.0) * math.log(self.gamma * (self.k - 1) + 1.0)

        self.which_phi1 = phi1
        self.which_phi2 = phi2
        self.which_psi = psi
        self.num_layers = num_layers

    def a_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        a(x,y) = 1/gamma + (1 - 1/gamma) * sigmoid( x·w_a + y·w_a )
        X: (k, d)
        A: (k, k)
        """
        s = X @ self.w_a
        Ssum = s[:, None] + s[None, :]
        A01 = torch.sigmoid(Ssum)
        return self.minv + self.range * A01

    def m1_vec(self, X: torch.Tensor) -> torch.Tensor:
        logits = X @ self.w_m1
        return torch.sigmoid(logits)

    def m2_vec(self, X: torch.Tensor) -> torch.Tensor:
        logits = X @ self.w_m2 + self.b_m2
        return F.softplus(logits)

    def phi1_one(self, t: torch.Tensor) -> torch.Tensor:
        """phi(t) = 1 - exp(-C * t), increasing; phi'' = -C * phi' <= -C * phi'."""
        return 1.0 - torch.exp(-self.C * torch.clamp(t, min=0.0))

    def phi1_two(self, t: torch.Tensor) -> torch.Tensor:
        """
        phi(t) = (-1/C) t^2 exp(-Ct) - (2/C^2 - 2a/C) t e^(-Ct)
                 - (b + a^2) e^(-Ct) / C + 2a e^(-Ct) / C^2
                 - 2 e^(-Ct) / C^3
        phi is increasing; phi'' <= - C * phi' in [0, 1].
        """
        a = 3
        b = 5
        exp = torch.exp(-self.C * t)
        return (
            (-1.0 / self.C) * t**2 * exp
            - (2 / self.C**2 - 2 * a / self.C) * t * exp
            - (b + a**2) * exp / self.C
            + 2 * a * exp / self.C**2
            - 2 * exp / self.C**3
        )

    @staticmethod
    def phi2_one(u: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.clamp(u, min=0.0))

    @staticmethod
    def phi2_two(u: torch.Tensor) -> torch.Tensor:
        """Concave, increasing, positive"""
        return 1 - torch.exp(-torch.clamp(u, min=0.0))

    @staticmethod
    def psi_one(u: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.clamp(u, min=0.0))

    @staticmethod
    def psi_two(u: torch.Tensor) -> torch.Tensor:
        """Concave, increasing, positive"""
        return 1 - torch.exp(-torch.clamp(u, min=0.0))

    def F(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        y = phi( sum_{x,y in S} a(x,y) + sum_{z in S} m1(z) ) + psi( sum_{z in S} m2(z) )
        X: (k, d), mask: (k,) bool
        """
        def zero(_: Tensor) -> Tensor:
            return torch.tensor([0])

        phi1: Callable[[Tensor], Tensor]
        phi2: Callable[[Tensor], Tensor]
        psi: Callable[[Tensor], Tensor]
        match self.which_phi1:
            case SynPhi1.One: phi1 = self.phi1_one
            case SynPhi1.Two: phi1 = self.phi1_two
            case None: phi1 = zero
        match self.which_phi2:
            case SynPhi2.One: phi2 = self.phi2_one
            case SynPhi2.Two: phi2 = self.phi2_two
            case None: phi2 = zero
        match self.which_psi:
            case SynPsi.One: psi = self.psi_one
            case SynPsi.Two: psi = self.psi_two
            case None: psi = zero

        y = torch.tensor([])
        for i in range(self.num_layers):
            M = (mask[:, None] & mask[None, :]).to(X.dtype)
            A = self.a_matrix(X)
            pair_sum = (A * M).sum()

            m1_vals = self.m1_vec(X)
            mod1_sum = (m1_vals * mask.to(X.dtype)).sum()

            m2_vals = self.m2_vec(X)
            mod2_sum = (m2_vals * mask.to(X.dtype)).sum()

            phi1_part = phi1(pair_sum)
            phi2_part = mod2_sum

            if i == 0:
                y = phi1_part + phi2_part
            else:
                y = psi(self.alpha * (phi1_part + phi2_part) + (1 - self.alpha) * y)

        return y


class SyntheticSetDataset(Dataset):
    """
    Elements ~ N(0, I_d), pad to k with zeros, mask True for first s
    Target y = phi( sum a + sum m1 ) + psi( sum m2 )
    """

    def __init__(
        self,
        N: int,
        d: int,
        k: int,
        gamma: float = 3.5,
        phi1: Optional[SynPhi1] = SynPhi1.One,
        phi2: Optional[SynPhi2] = SynPhi2.One,
        psi: Optional[SynPsi] = SynPsi.One,
        num_layers: int = 2,
    ):
        super().__init__()
        self.N = N
        self.d = d
        self.k = k
        self.fixed = FixedFunctions(
            d=d, k=k, gamma=gamma, phi1=phi1, phi2=phi2, psi=psi, num_layers=num_layers
        )

        Xs = torch.zeros(N, k, d, dtype=torch.float32)
        masks = torch.zeros(N, k, dtype=torch.bool)
        ys = torch.zeros(N, dtype=torch.float32)

        for n in range(N):
            s = torch.randint(low=1, high=k + 1, size=(1,)).item()
            elems = torch.rand(s, d, dtype=torch.float32)
            Xs[n, :s, :] = elems
            masks[n, :s] = True
            ys[n] = self.fixed.F(Xs[n], masks[n])

        self.X = Xs
        self.y = ys

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    
def get_train_val_test_int_dataset(device, num_features, total_number_of_elements_in_ground_set,
                                num_data_pts, max_set_size, data_type: str, folds = (0.7, 0.15, 0.15)):
    train_size = int(num_data_pts * folds[0])
    val_size = int(num_data_pts * folds[1])
    test_size = num_data_pts - train_size - val_size
    assert data_type[:-1] == "INT"
    num_layers = int(data_type[-1])
    full = SyntheticSetDataset(num_data_pts, num_features, max_set_size, num_layers=num_layers)
    g = torch.Generator()
    train_dataset, val_dataset, test_dataset = random_split(full, [train_size, val_size, test_size], generator=g)
    return train_dataset, val_dataset, test_dataset
