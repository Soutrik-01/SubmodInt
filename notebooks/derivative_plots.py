import os
from collections import defaultdict
from pathlib import Path

import marimo as mo
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam

from train_inference.train_test import Checkpoint, load_run

torch.serialization.add_safe_globals([
    Adam,
    Checkpoint,
    defaultdict,
    dict,
    getattr,
    torch.Generator,
])

device = torch.device("cuda")

B = 10
N = 8
k = 8

FLEXSUBNET_RUN = Path(os.environ["FLEXSUBNET_RUN"])
SUBMODINT_RUN = Path(os.environ["SUBMODINT_RUN"])

mo.md("# FlexSubNet")

flexsubnet_cli, flexsubnet = load_run(FLEXSUBNET_RUN)
d = flexsubnet_cli.dataset_opts.feature_dimension

print(nn.functional.linear(torch.rand(B, N, d), flexsubnet.log_weight).sum(dim=1))

def flexsubnet_plots():
    x = torch.linspace(50, 100, 1000).view(-1, 1).requires_grad_(True)

    flexsubnet.zero_grad()

    y = flexsubnet.OuterIntegrand_First_out(x)
    y.sum().backward()

    _, ax = plt.subplots()
    ax.plot(x.detach().numpy(), y.detach().numpy(), label="first derivative")
    ax.plot(x.detach().numpy(), x.grad.numpy(), label="second derivative")
    ax.legend()
    ax.set_ylim(-1e-8, 1e-8)
    plt.show()


flexsubnet_plots()

mo.md("# SubmodInt")

submodint_cli, submodint = load_run(SUBMODINT_RUN)
d = submodint_cli.dataset_opts.feature_dimension

x = torch.randn(B, N, d)

(
    submodint.interaction_models[0](
        x, x.unsqueeze(2).expand(B, N, N, d), x.unsqueeze(1).expand(B, N, N, d)
    )
    + submodint.m1s[0](x).sum(dim=1)
)

submodint.m2s[0](torch.randn(B, k, d)).sum(dim=1)

def submodint_plots():
    x = torch.linspace(0, 15, 1000).view(-1, 1).requires_grad_(True)

    phi1_1 = submodint.phi_inters[0]

    submodint.zero_grad()

    y = phi1_1(x)
    y.sum().backward()

    phi1_1_vals = y.detach().numpy()
    phi1_1_prime_vals = x.grad.numpy()

    _, ax = plt.subplots()
    ax.plot(x.detach().numpy(), phi1_1_vals, label=r"$\phi_1^{(1)}$")
    ax.plot(x.detach().numpy(), phi1_1_prime_vals, label=r"$\phi_1^{(1)}\prime$")
    ax.legend()
    plt.show()


submodint_plots()
