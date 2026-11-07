"""Parameter count of each model of configs/synthetic.toml, built from its table as scripts/run.py runs it (each
synthetic run also records it in its config.toml). From the repository root: PYTHONPATH=$PWD python
notebooks/param_count.py"""

import tomllib

import torch

from scripts.run import REPO, flags, resolve, tasks
from train_inference.train_test import Cli, create_model_and_optimizer

config = tomllib.loads((REPO / "configs/synthetic.toml").read_text())
for name, model, fixed, layers, datasets in tasks(config):
    dataset = (datasets or ["fac-loc"])[0]
    _, options = resolve(config, "synthetic", model, fixed, layers, dataset)
    cli = Cli.parse([options.pop("model"), dataset, *flags(options), "--seed", "0", "--run-dir", "-", "--log-dir", "-"])
    net, _ = create_model_and_optimizer(
        cli.model,
        cli.dataset_opts.feature_dimension,
        cli.dataset_opts.gamma,
        cli.dataset_opts.max_set_size,
        cli.submodint_opts,
        cli.baseline_opts,
        cli.train_opts.optimizer,
        cli.train_opts.learning_rate,
        cli.train_opts.weight_decay,
        torch.device("cpu"),
    )
    print(f"{name}: {sum(p.numel() for p in net.parameters())}")
