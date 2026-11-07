"""Downloads ImageNet-100 (clane9/imagenet-100), CIFAR-100 (uoft-cs/cifar100) and CLIP ViT-bigG-14 weights.

Usage: python download.py --data <DIR>
"""

import argparse

import open_clip
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="download directory")
DATA = ap.parse_args().data

snapshot_download("uoft-cs/cifar100", repo_type="dataset", local_dir=f"{DATA}/cifar100")
print("CIFAR-100 done", flush=True)
open_clip.create_model_and_transforms("ViT-bigG-14", pretrained="laion2b_s39b_b160k")
print("CLIP ViT-bigG-14 done", flush=True)
snapshot_download("clane9/imagenet-100", repo_type="dataset", local_dir=f"{DATA}/imagenet-100")
print("ImageNet-100 done", flush=True)
