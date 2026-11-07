"""Average submodularity ratio of trained set functions: the mean, over random S ⊊ T and e ∉ T drawn from the elements
of the run's test sets, of f(e | S) / f(e | T), where f(e | A) = f(A ∪ {e}) - f(A). Every ratio is at least 1 for a
submodular f. Sets are zero-padded to the dataset's maximum set size, as in training.

Usage: PYTHONPATH=$PWD python notebooks/submodularity_ratio.py RUN_DIR... [--samples N] [--seed SEED] [--device DEV]
where RUN_DIR is a run directory of train_inference/train_test.py, e.g. runs/synthetic/g-cut/SubmodInt-L2/seed0.
"""

import argparse
import math
from pathlib import Path

import torch

from models.baselines import subnet_flex_dual
from models.edsf.dsf import ExtendedDSF
from models.submodint import SubmodInt
from train_inference.train_test import create_datasets, load_run


def set_function(model, device):
    """f on a batch of zero-padded sets (B, N, d), called the way train_test.evaluate calls each model."""

    def f(X):
        if isinstance(model, subnet_flex_dual):
            return model(X)[0].squeeze(-1)
        if isinstance(model, (SubmodInt, ExtendedDSF)):
            return model(X).squeeze(-1)
        return model(X, torch.zeros(X.shape[0], device=device)).squeeze(-1)

    return f


def element_pool(cli, device):
    """The distinct elements of the run's test sets; the dataset is rebuilt from its seed, as in training."""
    torch.manual_seed(42)
    _, _, test_set = create_datasets(cli.dataset, device, cli.dataset_opts)
    rows = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(device).flatten(0, 1)
    return torch.unique(rows[rows.abs().sum(-1) > 0], dim=0)


@torch.no_grad()
def marginal_gains(f, pool, max_set_size, samples, batch_size=128):
    """f(e | S) and f(e | T) for `samples` random draws of S ⊊ T, |T| < max_set_size, and e ∉ T."""
    S, S_e, T, T_e = (torch.zeros(samples, max_set_size, pool.shape[1], device=pool.device) for _ in range(4))
    for i in range(samples):
        t = int(torch.randint(1, max_set_size, ()))
        s = int(torch.randint(0, t, ()))
        idx = torch.randperm(len(pool), device=pool.device)[: t + 1]
        T[i, :t] = T_e[i, :t] = pool[idx[:t]]
        S[i, :s] = S_e[i, :s] = pool[idx[:s]]
        S_e[i, s] = T_e[i, t] = pool[idx[t]]
    f_S, f_S_e, f_T, f_T_e = (torch.cat([f(x) for x in X.split(batch_size)]) for X in (S, S_e, T, T_e))
    return f_S_e - f_S, f_T_e - f_T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--samples", type=int, default=1000, help="number of draws of (S, T, e)")
    ap.add_argument("--min-gain", type=float, default=1e-6, help="draws with f(e | T) below this are left out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    args = ap.parse_args()

    for run_dir in args.run_dirs:
        cli, model = load_run(run_dir, args.device)
        model.eval()
        pool = element_pool(cli, args.device)
        torch.manual_seed(args.seed)
        gain_S, gain_T = marginal_gains(set_function(model, args.device), pool, cli.dataset_opts.max_set_size,
                                        args.samples)
        kept = gain_T > args.min_gain
        ratio = gain_S[kept] / gain_T[kept]
        print(f"{run_dir}: average ratio {ratio.mean():.4f} (standard error {ratio.std() / math.sqrt(len(ratio)):.4f})"
              f" over {len(ratio)} draws" + (f"; {int((~kept).sum())} draws with f(e | T) <= {args.min_gain} left out"
                                             if not kept.all() else ""))


if __name__ == "__main__":
    main()
