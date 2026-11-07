"""Compute cost of the models on random sets: median time and peak GPU memory of an inference forward pass and of a
training step (forward and backward), for batches of BATCH_SIZE sets of each size in SET_SIZES. Needs a GPU.

Usage: PYTHONPATH=$PWD python notebooks/compute_cost.py
"""

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import torch
import torch._dynamo
from torch import Tensor
from torch.nn.functional import mse_loss

from models.baselines import DSF, DeepSet, SetTransformer, subnet_flex_dual
from models.submodint import InteractionFn, SubmodInt, SubmodIntConfig

torch._dynamo.config.recompile_limit = 2**10
torch._dynamo.config.cache_size_limit = 2**10


@dataclass(slots=True)
class Result:
    name: str
    inf_time_ms: float
    train_time_ms: float
    inf_mem_mb: float
    train_mem_mb: float


BATCH_SIZE: Final = 128
SET_SIZES: Final = (8, 16, 32, 128, 256, 512, 1024, 2048, 4096)
D: Final = 3
DEVICE: Final = torch.device("cuda")
N_WARMUP: Final = 10
N_REPEAT: Final = 50
type Model = DSF | SetTransformer | DeepSet | subnet_flex_dual | SubmodInt


def make_models(set_size: int) -> list[tuple[str, Model]]:
    return [
        (
            "DeepSets",
            DeepSet(
                num_feature=D,
                in_d=3,
                hidden_layers=[50],
                dev=str(DEVICE),
            ).to(DEVICE),
        ),
        (
            "Set Transformer",
            SetTransformer(num_feature=D, in_d=3, hidden_layers=[50]).to(DEVICE),
        ),
        ("DSF", DSF(input_feature_dim=D, hidden_layers=[12], num_steps=2).to(DEVICE)),
        (
            "FlexSubNet",
            subnet_flex_dual(
                num_feature=D, in_d=3, hidden_layers=[50], num_rnn_steps=2, dev=str(DEVICE)
            ).to(DEVICE),
        ),
        *(
            (
                f"SubmodInt ({num_layers} layer{'s' if num_layers > 1 else ''})",
                SubmodInt.from_config(
                    feature_dim=D,
                    gamma=2.5,
                    max_set_size=set_size,
                    config=SubmodIntConfig(
                        num_layers=num_layers,
                        transform_hidden_dims=[32],
                        interaction_hidden_dims=[10, 10],
                        modular_hidden_dims=[32],
                        independent_params=False,
                        fix_continuous_transforms=False,
                        interaction_fn=InteractionFn.B,
                        no_m1=True,
                        no_m2=False,
                        no_phi_ccm=False,
                        no_outer_transform=False,
                        nb_steps=50,
                    ),
                ).to(DEVICE),
            )
            for num_layers in (1, 2, 5, 10)
        ),
    ]


def fwd_pass(model: Model, input: Tensor):
    match model:
        case SubmodInt() | subnet_flex_dual():
            _ = model(input)
        case _:
            _ = model(input, torch.zeros_like(input))


def fwd_and_bwd_pass(model: Model, input: Tensor, y: Tensor):
    match model:
        case SubmodInt():
            mse_loss(model(input).squeeze(-1), y).backward()
        case subnet_flex_dual():
            y_pred, a, b = model(input)
            loss = mse_loss(y_pred.squeeze(-1), y) + mse_loss(a, b)
            loss.backward()
        case _:
            mse_loss(model(input, torch.zeros_like(input)).squeeze(-1), y).backward()


def profile_fn[**P](
    func: Callable[P, None],
    setup: Callable[[], None] | None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> tuple[float, float]:
    """Return (median time_ms, peak mem_mb) over `N_REPEAT` timed runs."""
    for _ in range(N_WARMUP):
        if setup is not None:
            setup()
        func(*args, **kwargs)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    peak_mem_mb = 0.0
    for _ in range(N_REPEAT):
        if setup is not None:
            setup()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated()

        tic = time.monotonic_ns()
        func(*args, **kwargs)
        torch.cuda.synchronize()
        toc = time.monotonic_ns()

        times_ms.append((toc - tic) / 1e6)
        peak_mem_mb = max(
            peak_mem_mb,
            (torch.cuda.max_memory_allocated() - base_mem) / (2**20),
        )

    return statistics.median(times_ms), peak_mem_mb


def main():
    all_results: dict[int, list[Result]] = {}

    for set_size in SET_SIZES:
        input = torch.randn((BATCH_SIZE, set_size, D), device=DEVICE)
        y = torch.randn((BATCH_SIZE,), device=DEVICE)

        results: list[Result] = []

        for name, model in make_models(set_size):
            model.compile(dynamic=False)

            model.eval()
            with torch.inference_mode():
                inf_time_ms, inf_mem_mb = profile_fn(fwd_pass, None, model, input)

            model.train()
            train_time_ms, train_mem_mb = profile_fn(
                fwd_and_bwd_pass, model.zero_grad, model, input, y
            )
            model.zero_grad()

            results.append(
                Result(
                    name=name,
                    inf_time_ms=inf_time_ms,
                    train_time_ms=train_time_ms,
                    inf_mem_mb=inf_mem_mb,
                    train_mem_mb=train_mem_mb,
                )
            )
            del model

        all_results[set_size] = results

    model_names = [res.name for res in all_results[SET_SIZES[0]]]
    k_headers = " | ".join(f"k = {k}" for k in SET_SIZES)
    sep = "|---|" + "|".join("---" for _ in SET_SIZES) + "|"

    def print_table(title: str, fmt: Callable[[Result], str]):
        print(f"# {title}")
        print(f"| Model | {k_headers} |")
        print(sep)
        for i, name in enumerate(model_names):
            cells = " | ".join(fmt(all_results[k][i]) for k in SET_SIZES)
            print(f"| {name} | {cells} |")
        print()

    print_table("Inference Time (ms)", lambda r: f"{r.inf_time_ms:.3f}")
    print_table("Training Time (ms)", lambda r: f"{r.train_time_ms:.3f}")
    print_table("Inference Peak Mem (MiB)", lambda r: f"{r.inf_mem_mb:.2f}")
    print_table("Training Peak Mem (MiB)", lambda r: f"{r.train_mem_mb:.2f}")


if __name__ == "__main__":
    main()
