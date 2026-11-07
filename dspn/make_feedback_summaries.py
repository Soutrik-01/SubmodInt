"""Feedback summaries that --nnkmeans_feedback and --target_feedback load from $ROOT/<dset>/feedback_idx/:
NN-k-means summaries and stochastic-greedy summaries of the target FL, each of size set_size, on the
clean view of the training ground set.
"""

import argparse
import math
import os

import numpy as np
import torch

from submodular.fl import facilityLocation
from submodular.kmeans import nn_kmeans


def stochastic_greedy_fl(FL, k, generator, epsilon=0.01, chunk=1024):
    n = len(FL.V)
    covered = torch.zeros(n, device=FL.device)
    chosen = torch.zeros(n, dtype=torch.bool, device=FL.device)
    summary = []
    r = int(n / k * math.log(1 / epsilon))
    for _ in range(k):
        pool = (~chosen).nonzero().squeeze(1)
        cand = pool[torch.randperm(len(pool), generator=generator, device=FL.device)[:r]]
        gains = torch.cat([(FL.kernel(c) - covered).clamp(min=0).sum(1) for c in cand.split(chunk)])
        best = cand[gains.argmax()]
        covered = torch.maximum(covered, FL.kernel(best.view(1))[0])
        chosen[best] = True
        summary.append(best.item())
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dset", required=True)
    ap.add_argument("--kw", type=float, default=0.3)
    ap.add_argument("--set_size", type=int, default=100)
    ap.add_argument("--n_summaries", type=int, default=10)
    ap.add_argument("--seed", type=int, default=10)
    args = ap.parse_args()
    path = os.path.join(args.root, args.dset)
    X = torch.from_numpy(np.ascontiguousarray(np.load(f"{path}/dataset/X_train.npy", mmap_mode="r")[:, :, 0]))
    FL = facilityLocation(X, torch.device("cuda"), args={"fl_on_the_fly": True, "kw": args.kw})
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    os.makedirs(f"{path}/feedback_idx", exist_ok=True)

    nnk = np.stack([nn_kmeans(FL.V, args.set_size, generator=gen).cpu().numpy() for _ in range(args.n_summaries)])
    np.save(f"{path}/feedback_idx/nnkmeans_feedback_summary.npy", nnk)
    fl = np.stack([stochastic_greedy_fl(FL, args.set_size, gen) for _ in range(args.n_summaries)])
    np.save(f"{path}/feedback_idx/fl_feedback_summary.npy", fl)

    values = lambda S: FL.nested_fl(torch.from_numpy(S), [args.set_size])[0]
    random = np.stack([np.random.default_rng(i).choice(len(X), args.set_size, replace=False) for i in range(args.n_summaries)])
    print(f"FL/n of size-{args.set_size} summaries: target greedy {values(fl).mean():.4f}, NN-k-means {values(nnk).mean():.4f}, random {values(random).mean():.4f}")


if __name__ == "__main__":
    main()
