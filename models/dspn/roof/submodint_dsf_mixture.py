"""w SubmodInt(s) + (1 - w) DSF(s) on the sum s of a set's image features, with w in (0, 1) learned."""

import math

import torch
import torch.nn as nn


class SubmodIntDSFMixture(nn.Module):
    def __init__(self, submodint, dsf, w_init):
        super().__init__()
        self.submodint, self.dsf = submodint, dsf
        self.w_logit = nn.Parameter(torch.tensor(math.log(w_init / (1 - w_init))))

    def forward(self, s):
        w = torch.sigmoid(self.w_logit)
        return w * self.submodint(s) + (1 - w) * self.dsf(s)

    def project(self):
        """After each optimizer step: SubmodInt's own projection, and DSF's weights >= 0 as DSPN keeps them."""
        self.submodint.project()
        for p in self.dsf.parameters():
            p.data.clamp_(min=0.0)

    def abs_project(self):
        """At initialization, as DSPN starts DSF: its weights replaced by their absolute values."""
        self.submodint.project()
        for p in self.dsf.parameters():
            p.data = torch.abs(p.data)
