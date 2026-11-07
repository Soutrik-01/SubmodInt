"""Empirical submodularity check, in float64, of a trained DSPN whose set function takes the sum of the images'
features, f(S) = g(sum_{x in S} relu(features(x))).

Draws random nested sets A subset B (|B| < set_size) of the validation pool and an element v not in B, half the time
an exact copy of an element of B (the strongest interaction), and tests diminishing returns
f(B + v) - f(B) <= f(A + v) - f(A) and monotonicity f(A + v) >= f(A).
"""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from evaluate import load_model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--uid", required=True)
    ap.add_argument("--epoch", type=int, default=None, help="checkpoint epoch; default the last one")
    ap.add_argument("--root", required=True, help="root whose X_test is the validation pool")
    ap.add_argument("--tests", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--rtol", type=float, default=1e-9, help="violations below rtol * max|f| are rounding")
    ap.add_argument("--seed", type=int, default=0)
    cli = ap.parse_args()

    model, args, epoch = load_model(cli.run_dir, cli.uid, cli.epoch)
    if args["model_type"] in ("deepset", "set_transformer"):
        raise SystemExit(f"{args['model_type']} does not score a set through the sum of its features; nothing to check")
    model = model.double()
    X = torch.from_numpy(np.load(f"{cli.root}/{args['dset']}/dataset/X_test.npy")).double().cuda()
    emb = F.relu(model.feat(X)[1])
    f = lambda s: model.dsf(s).view(-1)
    n, k = len(emb), args["set_size"]
    g = torch.Generator(device="cuda").manual_seed(cli.seed)

    worst, n_viol, n_nonmono, f_max = 0.0, 0, 0, 0.0
    for start in range(0, cli.tests, cli.chunk):
        t = min(cli.chunk, cli.tests - start)
        order = torch.rand(t, n, device="cuda", generator=g).argsort(1)[:, :k]
        size_b = torch.randint(1, k - 1, (t,), device="cuda", generator=g)
        size_a = (torch.rand(t, device="cuda", generator=g) * size_b).long()
        prefix = torch.cat([torch.zeros(t, 1, emb.shape[1], dtype=emb.dtype, device="cuda"), emb[order].cumsum(1)], 1)
        rows = torch.arange(t, device="cuda")
        s_a, s_b = prefix[rows, size_a], prefix[rows, size_b]
        copy = order[rows, (torch.rand(t, device="cuda", generator=g) * size_b).long()]
        v = emb[torch.where(torch.rand(t, device="cuda", generator=g) < 0.5, copy, order[:, -1])]
        f_a, f_av, f_b, f_bv = f(s_a), f(s_a + v), f(s_b), f(s_b + v)
        f_max = max(f_max, float(f_bv.abs().max()))
        tol = cli.rtol * f_max
        excess = (f_bv - f_b) - (f_av - f_a)
        n_viol += int((excess > tol).sum())
        n_nonmono += int((f_av - f_a < -tol).sum())
        worst = max(worst, float(excess.max()))

    result = {"uid": cli.uid, "model_type": args["model_type"], "epoch": epoch, "tests": cli.tests,
              "violations": n_viol, "non_monotone": n_nonmono, "max_excess_gain": worst, "max_abs_f": f_max,
              "pass": n_viol == 0 and n_nonmono == 0}
    with open(f"{cli.run_dir}/trainlogs/{cli.uid}/submodularity_epoch{epoch}.json", "w") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
