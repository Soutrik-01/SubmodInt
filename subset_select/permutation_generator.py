import torch
import torch.nn as nn
import torch.nn.functional as F

from common.utils import *


def cudavar(av, x):
    """Adapt to CUDA or CUDA-less runs.  Annoying av arg may become
    useful for multi-GPU settings."""
    return (
        x.cuda()
        if (
            av.device == "cuda"
            or av.device == "cuda:0"
            or av.device == "cuda:1"
            or av.device == "cuda:2"
        )
        else x
    )


def pytorch_sample_gumbel(av, shape, eps=1e-20):
    U = cudavar(av, torch.rand(shape).float())
    return -torch.log(eps - torch.log(U + eps))


class PermutationGenerator_set(nn.Module):
    def __init__(self, av):
        super().__init__()
        MakeCopy(av, self)
        self.av = av
        self.max_set_size = self.max_len
        self.set_size_to_mask_map = [
            torch.cat((
                torch.repeat_interleave(
                    torch.tensor([1, 0]), torch.tensor([x, self.max_set_size - x])
                ).repeat(x, 1),
                torch.repeat_interleave(
                    torch.tensor([1, 0]), torch.tensor([0, self.max_set_size])
                ).repeat(self.max_set_size - x, 1),
            ))
            for x in range(1, self.max_set_size + 1)
        ]

        self.sets_maskB_all = cudavar(
            self.av,
            torch.stack([self.set_size_to_mask_map[x] == 0 for x in range(self.max_set_size)]),
        )

        self.latent_dim = self.av.PERM_NETWORK_LATENT_DIM
        self.output_dim = self.max_set_size
        self.linear1 = nn.Linear(self.num_features, self.latent_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(self.latent_dim, self.max_set_size)

    def pytorch_sinkhorn_iters_mask(self, log_alpha, maskB, temp, noise_factor=15.0, n_iters=20):
        batch_size = log_alpha.size()[0]
        n = log_alpha.size()[1]
        log_alpha = log_alpha.view(-1, n, n)
        noise = pytorch_sample_gumbel(self.av, [batch_size, n, n]) * noise_factor
        log_alpha = log_alpha + noise
        log_alpha = torch.div(log_alpha, temp)

        for i in range(n_iters):
            log_alpha = (
                log_alpha
                - (
                    torch.logsumexp(
                        log_alpha.masked_fill_(maskB, float("-inf")), dim=2, keepdim=True
                    )
                ).view(-1, n, 1)
            ).masked_fill_(maskB, float("-inf"))

            log_alpha = (
                log_alpha
                - (
                    torch.logsumexp(
                        log_alpha.masked_fill_(maskB, float("-inf")), dim=1, keepdim=True
                    )
                ).view(-1, 1, n)
            ).masked_fill_(maskB, float("-inf"))

        return torch.exp(log_alpha)

    def forward(self, U, S):
        av = self.av
        Universal_set_feat = cudavar(av, U)
        S_feat = Universal_set_feat[S]

        net = self.linear2(self.relu1(self.linear1(S_feat)))
        assert net.shape[0] == S_feat.shape[0]
        net_padded = cudavar(av, F.pad(net, pad=(0, 0, 0, net.shape[1] - net.shape[0]))).unsqueeze(
            0
        )

        output = self.pytorch_sinkhorn_iters_mask(
            net_padded,
            self.sets_maskB_all[net.shape[0] - 1],
            temp=1,
            noise_factor=1.0,
            n_iters=20,
        )
        Perm_mat = output.squeeze(0)[: net.shape[0], : net.shape[0]]
        P = Perm_mat

        return P
