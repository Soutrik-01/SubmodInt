"""Hold out a validation set from the training features, so that hyperparameters are never chosen on test data.

Writes dst_root/<dset>/dataset with X_train/Y_train = the training images minus per_class per class, and
X_test/Y_test = the held-out images (clean view only), plus the held-out indices. Every model is trained on
dst_root; evaluate.py on dst_root gives validation metrics and on src_root the test metrics.
"""

import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src_root", required=True)
ap.add_argument("--dst_root", required=True)
ap.add_argument("--dset", required=True)
ap.add_argument("--per_class", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

src, dst = f"{args.src_root}/{args.dset}/dataset", f"{args.dst_root}/{args.dset}/dataset"
os.makedirs(dst, exist_ok=True)
X, Y = np.load(f"{src}/X_train.npy", mmap_mode="r"), np.load(f"{src}/Y_train.npy")
rng = np.random.default_rng(args.seed)
val = np.sort(np.concatenate([rng.choice(np.nonzero(Y == c)[0], args.per_class, replace=False) for c in np.unique(Y)]))
train = np.setdiff1d(np.arange(len(Y)), val)
np.save(f"{dst}/val_idx.npy", val)
np.save(f"{dst}/X_test.npy", np.ascontiguousarray(X[val, :, 0]))
np.save(f"{dst}/Y_test.npy", Y[val])
np.save(f"{dst}/X_train.npy", X[train])
np.save(f"{dst}/Y_train.npy", Y[train])
print(f"{args.dset}: {len(train)} train, {len(val)} validation images -> {dst}")
