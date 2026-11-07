import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor

from models.neural_integral import ParallelNeuralIntegral, flatten_params

class Model(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(x) * torch.cos(torch.exp(x))

model = Model()
xs = torch.linspace(1, 5, 5000, requires_grad=True).unsqueeze(-1)
fig, ax = plt.subplots()
ax.plot(xs.detach(), (-torch.sin(torch.exp(xs))).detach())
ax.plot(
    xs.detach(),
    ParallelNeuralIntegral.apply(
        xs, torch.zeros_like(xs), model, flatten_params(model.parameters()), 50
    ).detach(),
)
