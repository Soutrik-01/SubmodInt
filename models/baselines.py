import math
from collections.abc import Sequence
from typing import Optional

import clap
import torch
import torch.nn as nn
import torch.nn.functional as F
from clap import arg, long
from serde import serde

from common.utils import generate_h
from models.mlp import MLP
from models.neural_integral import ParallelNeuralIntegral, flatten_params


@serde
@clap.group(title="Baseline config")
class BaselineConfig:
    """These options are used to configure the baseline models.

    Note that SetTransformer and DSF do not use --in-d. Also, --num-rnn-steps is
    only used by FlexSubNet and DSF."""
    in_d: int = arg(long, default_value=3)
    hidden_layers: list[int] = arg(long, default_value=[20, 20], num_args="*")
    num_rnn_steps: int = arg(long, default_value=2)
    dsf_readout: bool = arg(long)


class SetTransformer(nn.Module):
    def __init__(self, num_feature, in_d, hidden_layers, nb_steps=50, dev="cpu"):
        super().__init__()
        dim_input = 1
        num_inds = 2
        self.in_d = 1
        dim_hidden = hidden_layers[0]
        num_heads = 1
        num_outputs = 1
        dim_output = 1
        ln = False

        self.enc = nn.Sequential(ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln)).to(dev)
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            nn.Linear(dim_hidden, dim_output),
        ).to(dev)

    def forward(self, X, h):
        X = torch.min(X, dim=2)[0]
        X = X.unsqueeze(2)
        return self.dec(self.enc(X)).squeeze(-1).squeeze(-1)


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, key_mask=None):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        logits = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
        if key_mask is not None:
            pad = ~key_mask.repeat(self.num_heads, 1)[:, None, :]
            logits = logits.masked_fill(pad, float("-inf"))
        A = torch.softmax(logits, 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, mask=None):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, key_mask=mask)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, mask=None):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, key_mask=mask)


class MaskedSetTransformer(nn.Module):
    """Set Transformer on padded sets (B, T, d) with a (B, T) bool mask of real items -> (B,)."""

    def __init__(self, num_feature, hidden_layers):
        super().__init__()
        dim_hidden = hidden_layers[0]
        self.proj = nn.Linear(num_feature, dim_hidden)
        self.enc = ISAB(dim_hidden, dim_hidden, num_heads=1, num_inds=2)
        self.pool = PMA(dim_hidden, num_heads=1, num_seeds=1)
        self.out = nn.Linear(dim_hidden, 1)

    def forward(self, X, mask):
        return self.out(self.pool(self.enc(self.proj(X), mask), mask)).view(-1)


class ProjectedSetModel(nn.Module):
    """Linear front end on item embeddings, then a set model taking (X, mask=...) -> (B,).

    With raw_modular, adds the monotone modular term sum_i softplus(w . x_i) on the raw embeddings.
    With gate, scales the set model's output by a learnable gate that starts at 0 (kept >= 0 by training)."""

    def __init__(self, num_feature, proj_dim, set_model, raw_modular=False, gate=False):
        super().__init__()
        self.proj = nn.Linear(num_feature, proj_dim)
        self.set_model = set_model
        self.modular = nn.Linear(num_feature, 1) if raw_modular else None
        self.gate = nn.Parameter(torch.zeros(())) if gate else None

    def forward(self, X, mask):
        out = self.set_model(self.proj(X), mask=mask).reshape(-1)
        if getattr(self, "gate", None) is not None:
            out = self.gate * out
        if getattr(self, "modular", None) is not None:
            out = out + (F.softplus(self.modular(X)).squeeze(-1) * mask).sum(-1)
        return out


class DeepSet(nn.Module):
    def __init__(self, num_feature, in_d, hidden_layers, nb_steps=50, dev="cpu"):
        super().__init__()

        self.init_layer = nn.Linear(num_feature, in_d)
        self.num_feature = num_feature
        self.net = []
        self.in_d = in_d
        hs = [in_d] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:], strict=False):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()
        self.net = nn.Sequential(*self.net).to(dev)
        self.device = dev

    def forward(self, x, h):
        if len(list(x.shape)) == 2:
            x = x.unsqueeze(1)
        x = self.init_layer(x)
        x = torch.sum(x, dim=1)
        x = self.net(x)
        return x.squeeze(-1)


class IntegrandNN_zero_stable(nn.Module):
    def __init__(self, in_d, hidden_layers):
        super().__init__()
        self.net = []
        hs = [in_d] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:], strict=False):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()
        self.net.append(nn.ELU())
        self.net = nn.Sequential(*self.net)

        self.final_layer = nn.Linear(1, 1, bias=True)
        self.reset_params()

        self.relu = nn.ReLU()

    def reset_params(self):
        self.final_layer.weight.data = torch.abs(self.final_layer.weight.data)

    def forward(self, x, h):
        x = self.net(torch.cat((x, h), 1))
        return self.relu(self.final_layer(x))


class IntegrandNN(nn.Module):
    def __init__(self, in_d, hidden_layers):
        super().__init__()
        self.net = []
        hs = [in_d] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:], strict=False):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()
        self.net.append(nn.ELU())
        self.net = nn.Sequential(*self.net)

        self.final_layer = nn.Linear(5, 1, bias=False)
        self.reset_params()

        self.relu = nn.ReLU()

    def reset_params(self):
        self.final_layer.weight.data = torch.abs(self.final_layer.weight.data)

    def forward(self, x):
        x = self.net(torch.cat((x, torch.zeros(x.shape[0], 2, device=x.device)), 1)) + 1

        out = torch.cat((x**1.5, x**2, x**2.5, x**3, x**3.5), 1)
        return self.relu(self.final_layer(1 / (out + 1)))


class IntegrandNN_outer_level(nn.Module):
    def __init__(self, in_d, hidden_layers):
        super().__init__()
        self.net = []
        hs = [in_d] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:], strict=False):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()
        self.net.append(nn.ELU())
        self.net = nn.Sequential(*self.net)

        self.final_layer = nn.Linear(5, 1, bias=False)
        self.reset_params()

        self.relu = nn.ReLU()

    def reset_params(self):
        self.final_layer.weight.data = torch.abs(self.final_layer.weight.data)

    def forward(self, x):
        x = self.net(torch.cat((x, torch.zeros(x.shape[0], 2, device=x.device)), 1)) + 1

        out = torch.cat((x**1.1, x**1.1, x**1.5, x**2, x**2.5), 1)
        return self.relu(self.final_layer(1 / (out + 1)))


class neural_subm_two_level(nn.Module):
    def __init__(self, num_feature, in_d, hidden_layers, nb_steps=50, dev="cpu"):
        super().__init__()

        self.device = dev
        self.nb_steps = nb_steps
        self.OuterIntegrand = IntegrandNN(in_d, hidden_layers)
        self.InnerIntegrand = IntegrandNN(in_d, hidden_layers)

    def forward(self, x, h, xrandom=torch.tensor(0), hrandom=torch.tensor(0)):
        x = torch.sum(x, dim=1).unsqueeze(1)
        x0 = torch.zeros(x.shape).to(self.device)
        x = x.to(self.device)
        h = h.to(self.device)

        amax = torch.zeros(xrandom.shape).to(self.device) + 100
        hrandom = hrandom.to(self.device)
        xrandom = xrandom.to(self.device)

        OuterIntegralOut = ParallelNeuralIntegral.apply(
            x0,
            x,
            self.OuterIntegrand,
            flatten_params(self.OuterIntegrand.parameters()),
            h,
            self.nb_steps,
        )

        InnerIntegralOut = 0

        return OuterIntegralOut, InnerIntegralOut


class neural_subm_from_argmax(nn.Module):
    def __init__(
        self, neural_net, num_feature, in_d, hidden_layers, nb_steps=50, dev="cpu", mode="softmax"
    ):
        super().__init__()
        self.device = dev
        self.in_d = in_d
        if neural_net == deep_subm_baseline or neural_net == mixture_submodular:
            self.neural_net = neural_net(
                num_feature, in_d, hidden_layers, nb_steps=50, dev=self.device, task="summary"
            )
        else:
            self.neural_net = neural_net(
                num_feature, in_d, hidden_layers, nb_steps=50, dev=self.device
            )

        self.mode = mode
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def preprocess_input(self, UniversalSet, ArgMaxSet_Index, Not_ArgMax_Index):
        dim_element = self.neural_net.num_feature
        size_ArgMax = ArgMaxSet_Index.shape[0]
        size_Universe = UniversalSet.shape[0]

        Universe_Index = torch.tensor(list(range(size_Universe)))

        self.Upper_tri = torch.triu(torch.ones(size_ArgMax, size_Universe)).to(self.device)
        self.minus_large_upper_tri = -1000.0 * (1 - self.Upper_tri).to(self.device)

        self.minus_large_upper_tri_tensor = self.minus_large_upper_tri.unsqueeze(2).expand(
            size_ArgMax, size_Universe, dim_element
        )
        self.Upper_tri_tensor = self.Upper_tri.unsqueeze(2).expand(
            size_ArgMax, size_Universe, dim_element
        )

        self.ArgMaxSet_feat = UniversalSet[ArgMaxSet_Index, :]
        self.Not_ArgMaxSet_feat = UniversalSet[Not_ArgMax_Index, :]

        self.ArgMaxSet_Index = ArgMaxSet_Index
        self.Universe_Index = Universe_Index

        self.size_ArgMax = size_ArgMax
        self.size_Universe = size_Universe
        self.dim_element = dim_element

    def process_input(self, P):
        Permuted_ArgMaxSet_feat = torch.matmul(P, self.ArgMaxSet_feat)
        feat_mat = torch.cat((Permuted_ArgMaxSet_feat, self.Not_ArgMaxSet_feat), 0)
        self.feat_tensor = feat_mat.unsqueeze(0).expand(
            self.size_ArgMax, self.size_Universe, self.dim_element
        )

        ArgMaxSet_tensor = Permuted_ArgMaxSet_feat.unsqueeze(1)
        ArgMaxS_shifted_by_one_element = torch.cat([
            0 * ArgMaxSet_tensor[0:1, :, :],
            ArgMaxSet_tensor,
        ])[:-1, :, :]

        ArgMaxS_expanded = ArgMaxS_shifted_by_one_element.expand(
            self.size_ArgMax, self.size_Universe, self.dim_element
        )
        ArgMaxS_cumsum = torch.cumsum(ArgMaxS_expanded, dim=0)
        Arg_of_Softmax = ArgMaxS_cumsum + self.feat_tensor

        self.Arg_of_Softmax_padded = Arg_of_Softmax

    def forward(self, P):
        self.process_input(P)
        n, m = self.Arg_of_Softmax_padded.shape[0], self.Arg_of_Softmax_padded.shape[1]
        x = self.Arg_of_Softmax_padded.view(-1, self.dim_element)
        h = generate_h(self.Arg_of_Softmax_padded.view(-1, self.dim_element), self.in_d - 1)

        out_ = self.neural_net(x, h)
        out_ = out_.view(n, m)
        if self.mode == "softmax":
            out = self.Upper_tri * out_ + self.minus_large_upper_tri
            self.out_padded = out
            self.out_mat = self.LogSoftmax(out)
            self.log_like = torch.sum(torch.diagonal(self.out_mat))

            return self.log_like, out_

    @torch.no_grad()
    def maximize(self, U, budget):
        UU = U.to(self.device)
        seq = []
        for i in range(budget):
            h = generate_h(UU, self.in_d - 1)
            out = self.neural_net(UU, h)
            out[seq] = torch.min(out) - 100
            ii = torch.argmax(out)
            UU = UU + UU[ii]
            seq.append(ii.item())

        return seq


class set_model_from_argmax(nn.Module):
    """neural_subm_from_argmax for models that score the padded set itself, not its feature sum."""

    def __init__(self, neural_net, dev="cpu", mode="softmax"):
        super().__init__()
        self.neural_net = neural_net
        self.device = dev
        self.mode = mode
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def preprocess_input(self, UniversalSet, ArgMaxSet_Index, Not_ArgMax_Index):
        self.ArgMaxSet_feat = UniversalSet[ArgMaxSet_Index, :]
        self.Not_ArgMaxSet_feat = UniversalSet[Not_ArgMax_Index, :]
        self.size_ArgMax = ArgMaxSet_Index.shape[0]
        self.size_Universe = UniversalSet.shape[0]
        self.Upper_tri = torch.triu(torch.ones(self.size_ArgMax, self.size_Universe)).to(
            self.device
        )
        self.minus_large_upper_tri = -1000.0 * (1 - self.Upper_tri)

    def forward(self, P):
        n, m = self.size_ArgMax, self.size_Universe
        prefix = torch.matmul(P, self.ArgMaxSet_feat)
        universe = torch.cat((prefix, self.Not_ArgMaxSet_feat), 0)
        step = torch.arange(n, device=prefix.device)
        prefix_slots = (step[None, :] < step[:, None]).float()[:, None, :, None]
        candidate_slot = (step[None, :] == step[:, None]).float()[:, None, :, None]
        sets = prefix[None, None] * prefix_slots + universe[None, :, None] * candidate_slot
        mask = (step[None, :] <= step[:, None])[:, None, :].expand(n, m, n)
        out_ = self.neural_net(sets.reshape(n * m, n, -1), mask.reshape(n * m, n)).view(n, m)
        if self.mode == "softmax":
            out = self.Upper_tri * out_ + self.minus_large_upper_tri
            self.out_mat = self.LogSoftmax(out)
            self.log_like = torch.sum(torch.diagonal(self.out_mat))
            return self.log_like, out_

    @torch.no_grad()
    def maximize(self, U, budget):
        UU = U.to(self.device)
        seq = []
        for _ in range(budget):
            k = len(seq) + 1
            sets = torch.zeros(UU.shape[0], k, UU.shape[1], device=self.device)
            if seq:
                sets[:, : k - 1] = UU[seq]
            sets[:, k - 1] = UU
            mask = torch.ones(UU.shape[0], k, dtype=torch.bool, device=self.device)
            scores = self.neural_net(sets, mask)
            scores[seq] = scores.min() - 100
            seq.append(int(torch.argmax(scores)))
        return seq


class mixture_submodular(nn.Module):
    def __init__(
        self, num_feature, in_d, hidden_layers, nb_steps=50, dev="cpu", num_rnn_steps=2, task=None
    ):
        super().__init__()

        self.task = task
        self.num_rnn_steps = num_rnn_steps
        self.num_feature = num_feature
        self.in_d = in_d
        self.device = dev
        self.relu = nn.ReLU()
        self.mixnet = nn.Linear(in_d, 1)

    def reset_parameters(self):
        self.log_weight.data.fill_(1)

    def compute_modular_function(self, x_, h_, n):
        x = x_

        if len(list(x.shape)) == 3:
            x = torch.sum(x, dim=[1, 2]).unsqueeze(1)
        else:
            x = torch.sum(x, dim=1).unsqueeze(1)

        x = x.to(self.device)
        h = h_.to(self.device)

        return x, h

    def forward(self, x_, h_):
        x, h = self.compute_modular_function(x_, h_, 0)

        comp = []
        for i in range(self.in_d):
            comp.append(x)
            if self.task != "summary":
                x = torch.log(x)

            if self.task == "summary":
                x = torch.log(torch.abs(x) + 1)

        OuterIntegralOut = self.mixnet(torch.cat(tuple(comp), 1))

        return OuterIntegralOut


class deep_subm_baseline(nn.Module):
    def __init__(
        self, num_feature, in_d, hidden_layers, nb_steps=50, dev="cpu", num_rnn_steps=2, task=None
    ):
        super().__init__()
        self.task = task
        self.num_rnn_steps = num_rnn_steps
        self.num_feature = num_feature

        self.device = dev
        self.in_d = in_d
        self.splus = nn.Softplus()

        self.log_weight = nn.Parameter(torch.Tensor(1, self.num_feature))

        self.thrs = nn.Parameter(torch.Tensor(1))
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        self.log_weight.data.fill_(1)
        self.thrs.data.fill_(1)

    def compute_modular_function(self, x_, h_, n):
        wt = self.log_weight

        x = nn.functional.linear(x_, wt, bias=None)

        if len(list(x.shape)) == 3:
            x = torch.sum(x, dim=[1, 2]).unsqueeze(1)
        else:
            x = torch.sum(x, dim=1).unsqueeze(1)

        x = x.to(self.device)
        h = h_.to(self.device)

        return x, h

    def forward(self, x_, h_):
        for n in range(self.num_rnn_steps):
            x, _ = self.compute_modular_function(x_, h_, n)

            if self.task != "summary":
                x = torch.log(x**2 + self.thrs**2)

            if self.task == "summary":
                x = torch.log(torch.abs(x) + 1)
        return x.squeeze(-1)


class subnet_flex_recurrent(nn.Module):
    def __init__(self, num_feature, in_d, hidden_layers, nb_steps=50, dev="cpu", num_rnn_steps=2):
        super().__init__()

        self.num_rnn_steps = num_rnn_steps
        self.num_feature = num_feature
        self.in_d = in_d
        self.device = dev
        self.nb_steps = nb_steps
        self.OuterIntegrand = InnerUMNN(
            num_feature, in_d, hidden_layers, nb_steps=50, dev=self.device
        )

        self.sigmoid = nn.Sigmoid()
        self.log_weight = nn.Parameter(torch.Tensor(1, self.num_feature))
        self.sigmoid_param = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self):
        self.log_weight.data.fill_(1)

    def compute_modular_function(self, x_, h_, n):

        x = nn.functional.linear(x_, self.log_weight, bias=None)

        if len(list(x.shape)) == 3:
            x = torch.sum(x, dim=[1, 2]).unsqueeze(1)
        else:
            x = torch.sum(x, dim=1).unsqueeze(1)

        x = x.to(self.device)
        h = h_.to(self.device)

        return x, h

    def forward(self, x_, mask, h_):
        x_ = x_ * mask.unsqueeze(-1)
        OuterIntegralOut = 0
        for n in range(self.num_rnn_steps):
            x, h = self.compute_modular_function(x_, h_, n)

            x0 = torch.zeros(x.shape).to(self.device)

            lam = self.sigmoid(self.sigmoid_param)
            x = (1 - lam) * x + lam * OuterIntegralOut

            OuterIntegralOut = ParallelNeuralIntegral.apply(
                x0,
                x,
                self.OuterIntegrand,
                flatten_params(self.OuterIntegrand.parameters()),
                h,
                self.nb_steps,
            )

        return OuterIntegralOut


class modular_func(nn.Module):
    def __init__(self, num_feature, in_d, hidden_layers, nb_steps=50, dev="cpu"):
        super().__init__()
        self.num_feature = num_feature
        self.in_d = in_d
        self.device = dev
        self.nb_steps = nb_steps
        self.OuterIntegrand = InnerUMNN(
            num_feature, in_d, hidden_layers, nb_steps=50, dev=self.device
        )
        self.offset = torch.nn.Linear(1, 1, bias=True).to(self.device)
        self.splus = nn.Softplus()
        self.log_weight = nn.Parameter(torch.Tensor(1, self.num_feature))
        self.reset_parameters()

    def reset_parameters(self):
        self.log_weight.data.fill_(1)

    def forward(self, x, h):
        x = nn.functional.linear(x, self.log_weight, bias=None)

        if len(list(x.shape)) == 3:
            x = torch.sum(x, dim=[1, 2]).unsqueeze(1)
        else:
            x = torch.sum(x, dim=1).unsqueeze(1)

        x0 = torch.zeros(x.shape).to(self.device)
        x = x.to(self.device)
        return x


class neural_subm_one_level(nn.Module):
    def __init__(self, num_feature, in_d, hidden_layers, nb_steps=50, dev="cpu"):
        super().__init__()
        self.num_feature = num_feature
        self.in_d = in_d
        self.device = dev
        self.nb_steps = nb_steps
        self.OuterIntegrand = InnerUMNN(
            num_feature, in_d, hidden_layers, nb_steps=50, dev=self.device
        )
        self.offset = torch.nn.Linear(1, 1, bias=True).to(self.device)
        self.splus = nn.Softplus()
        self.log_weight = nn.Parameter(torch.Tensor(1, self.num_feature))
        self.reset_parameters()

    def reset_parameters(self):
        self.log_weight.data.fill_(1)

    def forward(self, x, h):
        x = nn.functional.linear(x, self.log_weight, bias=None)

        if len(list(x.shape)) == 3:
            x = torch.sum(x, dim=[1, 2]).unsqueeze(1)
        else:
            x = torch.sum(x, dim=1).unsqueeze(1)

        x0 = torch.zeros(x.shape).to(self.device)
        x = x.to(self.device)

        OuterIntegralOut = ParallelNeuralIntegral.apply(
            x0,
            x,
            self.OuterIntegrand,
            flatten_params(self.OuterIntegrand.parameters()),
            self.nb_steps,
        )

        return OuterIntegralOut


class InnerUMNN(nn.Module):
    def __init__(self, num_feature, in_d, hidden_layers, nb_steps=50, dev="cpu"):
        super().__init__()

        self.device = dev
        self.nb_steps = nb_steps
        self.Integrand = IntegrandNN(in_d, hidden_layers)

    def forward(self, x):
        xmax = torch.max(x) + 10
        xfinal = torch.zeros(x.shape).to(self.device) + xmax
        x = x.to(self.device)
        OuterIntegralOut = ParallelNeuralIntegral.apply(
            x, xfinal, self.Integrand, flatten_params(self.Integrand.parameters()), self.nb_steps
        )

        return OuterIntegralOut


class InnerUMNN_neg(nn.Module):
    def __init__(self, num_feature, in_d, hidden_layers, nb_steps=50, dev="cpu"):
        super().__init__()

        self.device = dev
        self.nb_steps = nb_steps
        self.Integrand = IntegrandNN_zero_stable(in_d, hidden_layers)

    def forward(self, x, h):
        x = x.to(self.device)
        h = h.to(self.device)
        x0 = torch.zeros(x.shape).to(self.device)

        OuterIntegralOut = ParallelNeuralIntegral.apply(
            x0, x, self.Integrand, flatten_params(self.Integrand.parameters()), h, self.nb_steps
        )

        return OuterIntegralOut


class subnet_flex_dual_alpha(nn.Module):
    def __init__(
        self,
        num_feature,
        in_d,
        hidden_layers,
        nb_steps=50,
        dev="cpu",
        num_rnn_steps=2,
        alpha_val=0.9,
    ):
        super().__init__()
        self.method_name = "subnet_flex_dual_alpha"
        self.num_rnn_steps = num_rnn_steps
        self.num_feature = num_feature
        self.in_d = in_d
        self.device = dev
        self.nb_steps = nb_steps
        self.OuterIntegrand_First_out = IntegrandNN_outer_level(in_d, hidden_layers)
        self.OuterIntegrand_Second_Out = IntegrandNN(in_d, hidden_layers)
        self.alpha_val = alpha_val
        self.sigmoid = nn.Sigmoid()
        self.log_weight = nn.Parameter(torch.Tensor(1, self.num_feature))
        self.sigmoid_param = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self):
        self.log_weight.data.fill_(1)

    def compute_modular_function(self, x_, h_, n):

        x = nn.functional.linear(x_, self.log_weight, bias=None)

        if len(list(x.shape)) == 3:
            x = torch.sum(x, dim=[1, 2]).unsqueeze(1)
        else:
            x = torch.sum(x, dim=1).unsqueeze(1)

        x = x.to(self.device)
        h = h_.to(self.device)

        return x, h

    def forward(self, x_, mask, h_):
        """
        dphi/dx = gamma(x)
        dgamma/dx = F(x)
        """

        OuterIntegral_First_out = 0
        OuterIntegral_Second_Out = 0
        for n in range(self.num_rnn_steps):
            x, h = self.compute_modular_function(x_ * mask.unsqueeze(-1), h_, n)
            x_input_second_integral = x
            x_input_to_first_integrand = x
            x0 = torch.zeros(x.shape).to(self.device)

            lam = self.sigmoid(self.sigmoid_param)
            x = (1 - lam) * x + lam * OuterIntegral_First_out

            x_input_second_integral = (
                1 - lam
            ) * x_input_second_integral + lam * OuterIntegral_Second_Out

            output_from_first_integrand = self.OuterIntegrand_First_out(x_input_second_integral, h)

            OuterIntegral_First_out = ParallelNeuralIntegral.apply(
                x0,
                x,
                self.OuterIntegrand_First_out,
                flatten_params(self.OuterIntegrand_First_out.parameters()),
                h,
                self.nb_steps,
            )
            xmax = torch.max(x_input_second_integral) + 10
            xfinal = torch.zeros(x.shape).to(self.device) + xmax
            OuterIntegral_Second_Out = (1 / self.alpha_val) * ParallelNeuralIntegral.apply(
                x_input_second_integral,
                xfinal,
                self.OuterIntegrand_Second_Out,
                flatten_params(self.OuterIntegrand_Second_Out.parameters()),
                h,
                self.nb_steps,
            )

        return OuterIntegral_First_out, output_from_first_integrand, OuterIntegral_Second_Out


class subnet_flex_dual(nn.Module):
    def __init__(self, num_feature, in_d, hidden_layers, nb_steps=50, dev="cpu", num_rnn_steps=2):
        super().__init__()
        self.method_name = "subnet_flex_dual"
        self.num_rnn_steps = num_rnn_steps
        self.num_feature = num_feature
        self.in_d = in_d
        self.device = dev
        self.nb_steps = nb_steps
        self.OuterIntegrand_First_out = IntegrandNN_outer_level(in_d, hidden_layers)
        self.OuterIntegrand_Second_Out = IntegrandNN(in_d, hidden_layers)

        self.sigmoid = nn.Sigmoid()
        self.log_weight = nn.Parameter(torch.Tensor(1, self.num_feature))
        self.sigmoid_param = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self):
        self.log_weight.data.fill_(1)

    def compute_modular_function(self, x_):
        return nn.functional.linear(x_, self.log_weight, bias=None).sum(1)

    def forward(self, x_):
        """
        dphi/dx = gamma(x)
        dgamma/dx = F(x)
        """

        OuterIntegral_First_out = 0
        OuterIntegral_Second_Out = 0
        for _ in range(self.num_rnn_steps):
            x = self.compute_modular_function(x_)
            x_input_second_integral = x
            x0 = torch.zeros(x.shape).to(self.device)

            lam = self.sigmoid(self.sigmoid_param)
            x = (1 - lam) * x + lam * OuterIntegral_First_out

            x_input_second_integral = (
                1 - lam
            ) * x_input_second_integral + lam * OuterIntegral_Second_Out

            output_from_first_integrand = self.OuterIntegrand_First_out(x_input_second_integral)

            OuterIntegral_First_out = ParallelNeuralIntegral.apply(
                x0,
                x,
                self.OuterIntegrand_First_out,
                flatten_params(self.OuterIntegrand_First_out.parameters()),
                self.nb_steps,
            )
            xmax = torch.max(x_input_second_integral) + 10
            xfinal = torch.zeros(x.shape).to(self.device) + xmax
            OuterIntegral_Second_Out = ParallelNeuralIntegral.apply(
                x_input_second_integral,
                xfinal,
                self.OuterIntegrand_Second_Out,
                flatten_params(self.OuterIntegrand_Second_Out.parameters()),
                self.nb_steps,
            )
        return OuterIntegral_First_out, output_from_first_integrand, OuterIntegral_Second_Out


class subnet_flex_dual_non_monotone(nn.Module):
    def __init__(
        self,
        num_feature,
        in_d,
        hidden_layers,
        nb_steps=50,
        dev="cpu",
        num_rnn_steps=1,
        thrs=45000,
        two_integrals=True,
    ):
        super().__init__()
        self.method_name = "subnet_flex_dual_non_monotone"
        self.num_rnn_steps = num_rnn_steps
        self.num_feature = num_feature
        self.in_d = in_d
        self.device = dev
        self.nb_steps = nb_steps
        self.first_subnet = subnet_flex_dual(
            num_feature, in_d, hidden_layers, nb_steps=50, dev=self.device, num_rnn_steps=1
        )
        self.second_subnet = subnet_flex_dual(
            num_feature, in_d, hidden_layers, nb_steps=50, dev=self.device, num_rnn_steps=1
        )
        self.relu = nn.ReLU()
        self.bias = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()
        self.thrs = thrs
        self.two_integrals = two_integrals
        self.offset = 0.8

    def reset_parameters(self):
        self.bias.data.fill_(6.0)

    def rebatch(self, x, h):
        x_sum = torch.sum(x, dim=[1, 2])
        true = x_sum > self.thrs

        a = torch.nonzero(true.int()).squeeze(1)
        b = torch.nonzero(1 - true.int()).squeeze(1)

        x_less_ths = x[b]
        x_higher_ths = x[a]
        return x_less_ths, x_higher_ths, h[b], h[a], b, a

    def forward(self, x_, h_):
        """
        dphi/dx = gamma(x)
        dgamma/dx = F(x)
        """
        x_less_ths, x_higher_ths, h_1, h_2, b, a = self.rebatch(x_, h_)

        y_pred = torch.empty(x_.shape[0], 1).to(self.device)
        gamma_x = torch.empty(x_.shape[0], 1).to(self.device)
        F_x = torch.empty(x_.shape[0], 1).to(self.device)

        if x_less_ths.shape[0] > 0:
            y_pred_1, gamma_x_1, F_x_1 = self.first_subnet(x_less_ths, h_1)
            y_pred[b] = y_pred_1
            gamma_x[b] = gamma_x_1
            F_x[b] = F_x_1

        if x_higher_ths.shape[0] > 0:
            if self.two_integrals == True:
                y_pred_2, gamma_x_2, F_x_2 = self.second_subnet(0.8 - x_higher_ths, h_2)
            else:
                y_pred_2, gamma_x_2, F_x_2 = self.first_subnet(0.8 - x_higher_ths, h_2)

            y_pred[a] = y_pred_2
            gamma_x[a] = gamma_x_2
            F_x[a] = F_x_2

        return y_pred, gamma_x, F_x


class subnet_flex_dual_non_monotone_switch(nn.Module):
    def __init__(self, num_feature, in_d, hidden_layers, nb_steps=50, dev="cpu", num_rnn_steps=1):
        super().__init__()
        self.method_name = "subnet_flex_dual_non_monotone"
        self.num_rnn_steps = num_rnn_steps
        self.num_feature = num_feature
        self.in_d = in_d
        self.device = dev
        self.nb_steps = nb_steps
        self.first_subnet = subnet_flex_dual(
            num_feature, in_d, hidden_layers, nb_steps=50, dev=self.device, num_rnn_steps=1
        )
        self.second_subnet = subnet_flex_dual(
            num_feature, in_d, hidden_layers, nb_steps=50, dev=self.device, num_rnn_steps=1
        )
        self.relu = nn.ReLU()
        self.shift_2c = nn.Parameter(torch.Tensor(1))
        self.param_1 = nn.Parameter(torch.Tensor(1))
        self.param_2 = nn.Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        self.shift_2c.data.fill_(0.8)
        self.param_1.data.fill_(1)
        self.param_2.data.fill_(0.7)

    def rebatch(self, x, h):
        x_sum = torch.sum(x, dim=[1, 2])
        true = x_sum > 45000

        a = torch.nonzero(true.int()).squeeze(1)
        b = torch.nonzero(1 - true.int()).squeeze(1)

        x_less_ths = x[b]
        x_higher_ths = x[a]
        return x_less_ths, x_higher_ths, h[b], h[a], b, a

    def forward(self, x_, h_):
        """
        dphi/dx = gamma(x)
        dgamma/dx = F(x)
        """
        x_sum = torch.sum(x_, dim=[1, 2])
        switch = torch.zeros(x_sum.shape[0], 1).to(self.device)
        switch[45000 - x_sum <= 0] = 0
        switch[45000 - x_sum > 0] = 1
        y_pred, gamma_x, F_x = tuple(
            self.first_subnet(x_, h_)[i] * switch
            + (1 - switch) * self.relu(self.second_subnet(self.shift_2c - x_, h_)[i])
            for i in range(3)
        )
        return y_pred, gamma_x, F_x


class DSF(nn.Module):
    """
    Deep Submodular Function (DSF) with channel-as-set hierarchy.

    Representation at DSF step i:
        H_i has shape (B, m_i, d_i)

    Each DSF step i:
      - has m_i modular networks (heads) g_{i,k}: R^{d_{i-1}} -> R^{d_i}
      - each head is applied pointwise to the m_{i-1} "elements" (axis=1) of H_{i-1}
      - sum over axis=1 (set aggregation) -> (B, 1, d_i) per head
      - concat heads -> (B, m_i, d_i)
      - apply concave transform log1p elementwise

    Inputs:
      - x: (B, N, input_feature_dim)  (N can be variable across batches)
    Output:
      - (B, 1)

    Notes:
      - To keep log1p well-defined, we ensure nonnegativity before log1p using a final nonnegative activation.
      - "hidden_layers" is reused for every modular MLP in every DSF step.
    """

    def __init__(
        self,
        input_feature_dim: int,
        hidden_layers: Sequence[int],
        num_steps: int,
        feature_sizes: Optional[Sequence[int]] = None,
        out_dim_per_step: Optional[Sequence[int]] = None,
        readout: bool = False,
    ):
        super().__init__()

        if num_steps <= 0:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        if feature_sizes is not None and len(feature_sizes) != num_steps:
            raise ValueError(
                f"feature_sizes must have length num_steps ({num_steps}), got {len(feature_sizes)}"
            )
        if feature_sizes is not None and feature_sizes[-1] != 1 and not readout:
            raise ValueError(f"Final feature size must be 1 for output (B,1), got {feature_sizes[-1]}")

        self.input_feature_dim = int(input_feature_dim)
        self.hidden_layers = list(hidden_layers)
        for i in range(len(self.hidden_layers)):
            self.hidden_layers[i] = math.ceil(self.hidden_layers[i] / num_steps)
        self.num_steps = int(num_steps)

        if out_dim_per_step is None:
            out_dim_per_step = [self.input_feature_dim] * (num_steps - 1) + [self.input_feature_dim if readout else 1]
        else:
            if len(out_dim_per_step) != num_steps:
                raise ValueError(
                    f"out_dim_per_step must have length num_steps ({num_steps}), got {len(out_dim_per_step)}"
                )
            out_dim_per_step = list(map(int, out_dim_per_step))
            if out_dim_per_step[-1] != 1 and not readout:
                raise ValueError(f"Final output dim must be 1, got {out_dim_per_step[-1]}")

        if feature_sizes is None:
            feature_sizes = [8] * (num_steps - 1) + [8 if readout else 1]
        else:
            if len(feature_sizes) != num_steps:
                raise ValueError(
                    f"feature sizes must have length num_steps ({num_steps}), got {len(feature_sizes)}"
                )
            feature_sizes = list(map(int, feature_sizes))
            if feature_sizes[-1] != 1 and not readout:
                raise ValueError(f"Final output dim must be 1, got {out_dim_per_step[-1]}")

        self.feature_sizes = list(map(int, feature_sizes))
        self.out_dims = out_dim_per_step
        self.concave_fn = torch.log1p
        self.nonneg_fn = nn.ReLU

        steps = []
        d_prev = self.input_feature_dim
        for i in range(num_steps):
            m_i = self.feature_sizes[i]
            d_i = self.out_dims[i]
            steps.append(_DSFStep(d_in=d_prev, d_out=d_i, m_out=m_i, hidden_layers=self.hidden_layers,
                                  nonneg_fn=self.nonneg_fn))
            d_prev = d_i
        self.steps = nn.ModuleList(steps)

        self.readout = None
        if readout:
            n = self.feature_sizes[-1] * self.out_dims[-1]
            self.readout = nn.Parameter(torch.full((n,), math.log(math.expm1(1 / n))))

    def forward(self, x: torch.Tensor, h: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, N, input_feature_dim)
        returns: (B, 1)
        """

        if x.dim() != 3:
            raise ValueError(f"Expected input x to have shape (B,N,d), got {tuple(x.shape)}")
        if x.size(-1) != self.input_feature_dim:
            raise ValueError(
                f"Expected last dim {self.input_feature_dim}, got {x.size(-1)}"
            )

        H = x
        for i, step in enumerate(self.steps):
            S = step(H, mask if i == 0 else None)
            H = self.concave_fn(S)

        if self.readout is not None:
            return (F.softplus(self.readout) * H.flatten(1)).sum(dim=1, keepdim=True)
        if H.size(1) != 1 or H.size(2) != 1:
            raise RuntimeError(f"Internal error: final H shape expected (B,1,1), got {tuple(H.shape)}")
        return H.squeeze(-1)


class _DSFStep(nn.Module):
    """
    One DSF step: m_out heads, each head is an MLP d_in->d_out applied pointwise,
    then summed over the set axis (dim=1), then concatenated across heads.

    Input:  (B, m_in, d_in)
    Output: (B, m_out, d_out)  (pre-concave, nonnegative)
    """
    def __init__(
        self,
        d_in: int,
        d_out: int,
        m_out: int,
        hidden_layers: Sequence[int],
        nonneg_fn: type[nn.Module],
    ):
        super().__init__()
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.m_out = int(m_out)
        self.nonneg_fn = nonneg_fn

        self.heads = nn.ModuleList([
            MLP(
                in_dim=self.d_in,
                hidden_dims=hidden_layers,
                out_dim=self.d_out,
                final_act=nonneg_fn,
            )
            for _ in range(self.m_out)
        ])

    def forward(self, H: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if H.dim() != 3:
            raise ValueError(f"Expected H shape (B,m_in,d_in), got {tuple(H.shape)}")
        if H.size(-1) != self.d_in:
            raise ValueError(f"Expected last dim {self.d_in}, got {H.size(-1)}")

        outs = []
        for net in self.heads:
            Y = net(H)
            if mask is not None:
                Y = Y * mask.unsqueeze(-1).to(Y.dtype)
            S = Y.sum(dim=1, keepdim=True)
            outs.append(S)
        return torch.cat(outs, dim=1)
