"""Passive (E, M) pairs of dataset/generate_EM.py with sampling_type=[1, 0, 1, 0], written to
$ROOT/<dset>/passive_idx/passive_samples.pkl in the format utils.fetch_dataset_imgs_labels_passive_subsets reads.

Half the pairs are Style-I (M: a random set from one class, E: a random set from that class and another);
in the other half, M joins nearest-neighbour clumps from two classes and E is the NN-k-means summary of
those two classes. Distances use the clean CLIP view (the kernel in generate_EM.py is monotone in them).
"""

import argparse
import os
import pickle as pkl

import numpy as np
import torch
from tqdm import tqdm

from submodular.kmeans import nn_kmeans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dset", required=True)
    ap.add_argument("--n_pairs", type=int, default=40000, help="n_iteration * bsz of generate_EM.py")
    ap.add_argument("--set_size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=10)
    args = ap.parse_args()
    np.random.seed(args.seed)
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    path = os.path.join(args.root, args.dset)
    H = torch.from_numpy(np.ascontiguousarray(np.load(f"{path}/dataset/X_train.npy", mmap_mode="r")[:, :, 0])).cuda()
    Y = np.load(f"{path}/dataset/Y_train.npy")
    K = int(Y.max()) + 1
    by_class = [np.nonzero(Y == k)[0] for k in range(K)]
    dataset = {}

    def add(y_A, y_B, D_A, D_B):
        assert len(D_A) == len(D_B) == args.set_size
        dataset.setdefault((len(y_A), len(y_B)), []).append((y_A, y_B, np.asarray(D_A), np.asarray(D_B)))

    for _ in tqdm(range(args.n_pairs // 2), desc="Style-I"):
        c = np.random.choice(K)
        other = np.random.choice(np.delete(np.arange(K), c))
        D_A = np.random.choice(by_class[c], args.set_size, replace=False)
        D_B = np.random.choice(np.concatenate([by_class[c], by_class[other]]), args.set_size, replace=False)
        add(np.asarray([c]), np.asarray([c, other]), D_A, D_B)

    for _ in tqdm(range(args.n_pairs - args.n_pairs // 2), desc="clumps vs NN-k-means"):
        classes = np.random.choice(K, size=2, replace=False)
        D_A = []
        for c, num in zip(classes, [len(p) for p in np.array_split(np.arange(args.set_size), 2)]):
            members = by_class[c]
            seed_point = np.random.choice(members)
            d2 = ((H[members] - H[seed_point]) ** 2).sum(1)
            D_A.append(seed_point)
            D_A.extend(members[d2.topk(num, largest=False).indices[1:].cpu().numpy()])
        ground = np.concatenate([by_class[c] for c in sorted(classes)])
        D_B = ground[nn_kmeans(H[ground], args.set_size, generator=gen).cpu().numpy()]
        D_B = D_B[np.argsort(Y[D_B], kind="stable")]
        add(np.unique(Y[D_A]), np.unique(Y[D_B]), D_A, D_B)

    for key, rows in dataset.items():
        y_A, y_B, D_A, D_B = zip(*rows)
        dataset[key] = (y_A, y_B, torch.from_numpy(np.stack(D_A)), torch.from_numpy(np.stack(D_B)))
    os.makedirs(f"{path}/passive_idx", exist_ok=True)
    with open(f"{path}/passive_idx/passive_samples.pkl", "wb") as f:
        pkl.dump(dataset, f)
    print({k: len(v[2]) for k, v in dataset.items()})


if __name__ == "__main__":
    main()
