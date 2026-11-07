import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PosLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PosLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn((in_dim, out_dim)).abs_() * 0.1)
        
    def forward(self, x):
        return torch.matmul(x, self.weight)
    
    def relu(self):
        self.weight.relu_()


class MinComponent(nn.Module):
    def forward(self, x):
        return torch.min(x, dim=-1).values

    def relu(self):
        pass


class MiLU(nn.Module):
    def __init__(self, alpha : float = 1.0, inplace: bool = False):
        super(MiLU, self).__init__()
        self.inplace = inplace
        self.alpha = torch.tensor(alpha)

    def forward(self, input: Tensor) -> Tensor:
        return torch.minimum(input, self.alpha)


    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str