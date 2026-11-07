"""CLIP ViT-bigG-14 features in the layout DSPN expects: $ROOT/<dset>/dataset/{X,Y}_{train,test}.npy.

X_train is (N, 1280, 5): view 0 is the clean image (open_clip's validation transform), views 1-4 add
the augmentations of the authors' CLIP_embeddings.py (flip, rotation 5, colour distortion 0.1).
X_test is (N, 1280), clean only. Usage: python extract_clip.py --data <download.py's DIR> --root <ROOT> --dset IN100|CIFAR100
"""

import argparse
import glob
import io
import os

import numpy as np
import open_clip
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SOURCES = {
    "CIFAR100": ({"train": "cifar100/cifar100/train-*.parquet", "test": "cifar100/cifar100/test-*.parquet"}, "img", "fine_label"),
    "IN100": ({"train": "imagenet-100/data/train-*.parquet", "test": "imagenet-100/data/validation-*.parquet"}, "image", "label"),
}


class ParquetImages(Dataset):
    def __init__(self, data, pattern, image_col, label_col, transform):
        tables = [pq.read_table(f, columns=[image_col, label_col]) for f in sorted(glob.glob(f"{data}/{pattern}"))]
        self.images = pa.concat_arrays([t.column(image_col).combine_chunks().field("bytes").cast(pa.large_binary()) for t in tables])
        self.labels = np.concatenate([t.column(label_col).to_numpy() for t in tables])
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(Image.open(io.BytesIO(self.images[i].as_py())).convert("RGB"))


@torch.no_grad()
def encode(model, dataset, seed, bsz=500):
    loader = DataLoader(dataset, batch_size=bsz, num_workers=16, generator=torch.Generator().manual_seed(seed),
                        worker_init_fn=lambda w: torch.manual_seed(seed * 100 + w))
    out = []
    for i, x in enumerate(loader):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model.encode_image(x.cuda(non_blocking=True)).float().cpu())
        if i % 20 == 0:
            print(f"  batch {i}/{len(loader)}", flush=True)
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dset", choices=list(SOURCES), required=True)
    ap.add_argument("--data", required=True, help="the directory download.py wrote to")
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    patterns, image_col, label_col = SOURCES[args.dset]
    out = f"{args.root}/{args.dset}/dataset"
    os.makedirs(out, exist_ok=True)

    model, _, clean = open_clip.create_model_and_transforms("ViT-bigG-14", pretrained="laion2b_s39b_b160k")
    model = model.cuda().eval()
    s = 0.1
    augment = T.Compose(clean.transforms[:-1] + [
        T.RandomHorizontalFlip(), T.RandomRotation(5),
        T.RandomApply([T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8), T.RandomGrayscale(p=0.2),
    ] + clean.transforms[-1:])

    test = ParquetImages(args.data, patterns["test"], image_col, label_col, clean)
    if not os.path.exists(f"{out}/Y_test.npy"):
        np.save(f"{out}/X_test.npy", encode(model, test, seed=0))
        np.save(f"{out}/Y_test.npy", test.labels)
    print(f"{args.dset} test done: {len(test)}", flush=True)

    train = ParquetImages(args.data, patterns["train"], image_col, label_col, clean)
    for v in range(5):
        if not os.path.exists(f"{out}/X_train_view{v}.npy"):
            train.transform = clean if v == 0 else augment
            np.save(f"{out}/X_train_view{v}.npy", encode(model, train, seed=v))
        print(f"{args.dset} train view {v} done", flush=True)
    np.save(f"{out}/X_train.npy", np.stack([np.load(f"{out}/X_train_view{v}.npy") for v in range(5)], axis=-1))
    np.save(f"{out}/Y_train.npy", train.labels)
    for v in range(5):
        os.remove(f"{out}/X_train_view{v}.npy")
    print(f"{args.dset} train done: {len(train)}", flush=True)


if __name__ == "__main__":
    main()
