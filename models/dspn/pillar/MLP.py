'''ResNet in PyTorch.

Reference
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.feat = nn.Sequential(nn.Linear(1280, 1536), nn.ReLU(), 
                                nn.Linear(1536, 1792), nn.ReLU(),
                                nn.Linear(1792, 2048), nn.ReLU(),
                                )
    
    def forward(self, x, last=True):
        return None, self.feat(x)


class MLP_small(nn.Module):
    def __init__(self, input_dim=512, out_dim=512):
        super(MLP_small, self).__init__()
        self.feat = nn.Sequential(nn.Linear(input_dim, 1024), nn.ReLU(), 
                                nn.Linear(1024, 1024), nn.ReLU(),
                                nn.Linear(1024, out_dim), nn.ReLU(),
                                )
    
    def forward(self, x, last=True):
        return None, self.feat(x)

class SelfAttentionEncoder(nn.Module):
    def __init__(self):
        super(SelfAttentionEncoder, self).__init__()
        self.encoder = nn.Sequential(
            ISAB(dim_in=1280, dim_out=1536, num_heads=4, num_inds=10),
            ISAB(dim_in=1536, dim_out=1792, num_heads=4, num_inds=10),
            ISAB(dim_in=1792, dim_out=2048, num_heads=4, num_inds=10),
        )

    def forward(self, x, last=True):
        return None, self.encoder(x)