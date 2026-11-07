"""The DSPN paper's summarization metrics for a trained model, on the held-out split X_test/Y_test.

X_test is split per class into a pool (ground set V') and a probe test set, with a fixed seed.
FL Eval:  target FL(model's greedy summary of V') / target FL(target FL's greedy summary of V').
Offline:  probe accuracy after training on the model's greedy summary of an imbalanced copy of V'
          (every image of the class ranked r duplicated ceil(zipf_max / r) times).
Online:   the same with the summary from streaming maximization (submodular/optimizer.py's
          DSF_opt.streaming_max) over a class-incremental stream of that copy.
The probe is logistic regression on L2-normalized clean CLIP features.
"""

import argparse
import glob
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from dspn_network import DSPN
from submodular.fl import facilityLocation


def load_model(run_dir, uid, epoch):
    with open(f"{run_dir}/trainlogs/{uid}/arguments.json") as f:
        args = json.load(f)
    if epoch is None:
        epoch = max(int(p.split("_")[-1][:-3]) for p in glob.glob(f"{run_dir}/trainlogs/{uid}/dspn_ckpt_*.pt"))
    nesting = list(range(10, args["set_size"] + args["nesting_interval"], args["nesting_interval"]))
    out_dims = [2048] if args["dset"] in ["IN100", "CIFAR100"] else [512, 10]
    concave = ["relu"] * 4 if args["model_type"] == "deepset" else ["log1p", "exp", "sqrt", "linear"]
    model = DSPN(out_dims=out_dims, concave=concave, args=args, nesting_list=nesting)
    with torch.serialization.safe_globals([np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType]):
        state = torch.load(f"{run_dir}/trainlogs/{uid}/dspn_ckpt_{epoch}.pt", map_location="cpu")["net"]
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
    model.set_device(torch.device("cuda"))
    return model.cuda().eval(), args, epoch


class SetFunction:
    """f(S) - f(empty set) of a trained model on one ground set given by clean CLIP features X (n, d). The training
    losses only use differences of f, so the constant f(empty set) is arbitrary; the streaming algorithm assumes it
    is zero."""

    def __init__(self, model, X, per_element):
        self.model, self.X, self.per_element = model, X, per_element
        self.empty = 0.0
        if not per_element:
            self.emb = F.relu(model.feat(X)[1])
            self.empty = model.dsf(torch.zeros_like(self.emb[:1])).view(-1)

    def __call__(self, sets, chunk=512):
        """Values of the index sets, shape (len(sets),); sets is a (B, k) LongTensor."""
        if not self.per_element:
            return self.model.dsf(self.emb[sets].sum(1)).view(-1) - self.empty
        return torch.cat([self.model.dsf(F.relu(self.model.feat(self.X[s])[1])).view(-1) for s in sets.split(chunk)])

    def add_each(self, S, candidates):
        """f(S + {c}) for every candidate c."""
        if not self.per_element:
            base = self.emb[S].sum(0) if S else torch.zeros_like(self.emb[0])
            return self.model.dsf(base + self.emb[candidates]).view(-1) - self.empty
        sets = torch.cat([torch.tensor(S, dtype=torch.long, device="cuda").expand(len(candidates), -1), candidates.view(-1, 1)], 1)
        return self(sets, chunk=max(1, 50000 // sets.shape[1]))


@torch.no_grad()
def greedy(f, n, k):
    S, taken = [], torch.zeros(n, dtype=torch.bool, device="cuda")
    for _ in range(k):
        values = f.add_each(S, torch.arange(n, device="cuda"))
        values[taken] = -math.inf
        S.append(int(values.argmax()))
        taken[S[-1]] = True
    return S


@torch.no_grad()
def streaming(f, stream, K, L, delta, C=1.0, alpha=1e-5):
    """DSF_opt.streaming_max (Lavania et al., SDM 2020) for any set function."""
    S, cur, gains = [], 0.0, []
    for v in stream:
        with_v = f(torch.tensor([S + [v]], device="cuda")).item()
        g = with_v - cur
        gains.append(g)
        if len(S) < K:
            if g > alpha:
                S, cur = S + [v], with_v
            continue
        swaps = torch.tensor([[x for x in S if x != j] + [v] for j in S], device="cuda")
        values = f(swaps)
        j = int(values.argmax())
        if values[j].item() - cur >= C * cur / K or g >= np.mean(gains[-L:]) + delta:
            S, cur = [x for x in S if x != S[j]] + [v], values[j].item()
    return S


def probe(X_train, y_train, X_test, y_test, C):
    if len(np.unique(y_train)) < 2:
        return float((y_test == y_train[0]).mean()) if len(y_train) else 0.0
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(F.normalize(X_train, dim=1).cpu().numpy(), y_train)
    return float((clf.predict(F.normalize(X_test, dim=1).cpu().numpy()) == y_test).mean())


@torch.no_grad()
def evaluate_seed(model, args, X, Y, cli, seed):
    """One evaluation: per-class pool/probe-test split, class ranks of the imbalanced pool and stream order from seed."""
    rng = np.random.default_rng(seed)
    pool, test = [], []
    for c in np.unique(Y):
        idx = rng.permutation(np.nonzero(Y == c)[0])
        pool.append(idx[: len(idx) // 2])
        test.append(idx[len(idx) // 2:])
    pool, test = np.concatenate(pool), np.concatenate(test)
    per_element = args["model_type"] == "set_transformer"
    k = cli.budget

    FL = facilityLocation(X[pool].cpu(), torch.device("cuda"), args={"fl_on_the_fly": True, "kw": args["kw"]})
    f = SetFunction(model, X[pool], per_element)
    S_model = greedy(f, len(pool), k)
    covered, S_target = torch.zeros(len(pool), device="cuda"), []
    for _ in range(k):
        gains = torch.cat([(FL.kernel(c) - covered).clamp(min=0).sum(1) for c in torch.arange(len(pool), device="cuda").split(1024)])
        gains[S_target] = -math.inf
        S_target.append(int(gains.argmax()))
        covered = torch.maximum(covered, FL.kernel(torch.tensor([S_target[-1]], device="cuda"))[0])
    budgets = list(range(10, k + 1, 10))
    fl_model = FL.nested_fl(torch.tensor([S_model]), budgets)[:, 0]
    fl_target = FL.nested_fl(torch.tensor([S_target]), budgets)[:, 0]
    fl_eval = {b: round(float(m / t), 4) for b, m, t in zip(budgets, fl_model, fl_target)}

    ranks = dict(zip(rng.permutation(np.unique(Y)), range(1, len(np.unique(Y)) + 1)))
    dup = np.concatenate([[i] * math.ceil(cli.zipf_max / ranks[Y[i]]) for i in pool])
    f_dup = SetFunction(model, X[dup], per_element)
    S_offline = dup[greedy(f_dup, len(dup), k)]
    order = np.concatenate([rng.permutation(np.nonzero(Y[dup] == c)[0]) for c in rng.permutation(np.unique(Y))])
    S_online = dup[np.array(streaming(f_dup, order.tolist(), k, cli.stream_L, cli.stream_delta), dtype=np.int64)]

    acc = lambda S: probe(X[S], Y[S], X[test], Y[test], cli.probe_C)
    random_S = rng.choice(dup, k, replace=False)
    return {
        "seed": seed, "fl_eval": fl_eval[k], "offline_acc": acc(S_offline), "online_acc": acc(S_online),
        "online_summary_size": len(S_online), "fl_eval_by_budget": fl_eval, "empty_set_value": float(f.empty),
        "distinct_images": {"offline": len(set(S_offline)), "online": len(set(S_online))},
        "reference": {"target_fl_dedup_offline_acc": acc(pool[S_target]), "random_offline_acc": acc(random_S)},
        "setup": {"pool": len(pool), "imbalanced_pool": len(dup), "probe_test": len(test)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="directory training ran in (holds trainlogs/<uid>)")
    ap.add_argument("--uid", required=True)
    ap.add_argument("--epoch", type=int, default=None, help="checkpoint epoch; default the last one")
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="test", help="name of the split in --root's X_test (val or test), used in the output file name")
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--zipf_max", type=int, default=50)
    ap.add_argument("--stream_L", type=int, default=100)
    ap.add_argument("--stream_delta", type=float, default=0.0)
    ap.add_argument("--probe_C", type=float, default=100.0, help="chosen on train features")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_seeds", type=int, default=5, help="metrics are means over seeds seed, ..., seed + n_seeds - 1")
    cli = ap.parse_args()

    if os.environ.get("DSPN_GPU_MEMORY_FRACTION"):
        torch.cuda.set_per_process_memory_fraction(float(os.environ["DSPN_GPU_MEMORY_FRACTION"]))
    model, args, epoch = load_model(cli.run_dir, cli.uid, cli.epoch)
    path = os.path.join(cli.root, args["dset"])
    X = torch.from_numpy(np.load(f"{path}/dataset/X_test.npy")).float().cuda()
    Y = np.load(f"{path}/dataset/Y_test.npy")
    per_seed = [evaluate_seed(model, args, X, Y, cli, s) for s in range(cli.seed, cli.seed + cli.n_seeds)]
    metrics = ("fl_eval", "offline_acc", "online_acc")
    result = {
        "model_type": args["model_type"], "uid": cli.uid, "epoch": epoch, "dset": args["dset"],
        **{m: float(np.mean([r[m] for r in per_seed])) for m in metrics},
        "std": {m: float(np.std([r[m] for r in per_seed])) for m in metrics},
        "per_seed": per_seed, "args": vars(cli),
    }
    out = f"{cli.run_dir}/trainlogs/{cli.uid}/eval_{cli.split}_epoch{epoch}.json"
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps({k: result[k] for k in ("model_type", "epoch", *metrics, "std")}))


if __name__ == "__main__":
    main()
