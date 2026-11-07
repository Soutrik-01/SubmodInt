"""Trains and evaluates one model of the paper on one dataset, with its settings in configs/EXPERIMENT.toml.

    python scripts/run.py synthetic --model SubmodInt --dataset g-cut [--layers L] [--fixed] [--seed SEED]
    python scripts/run.py synthetic --model SubmodInt --dataset log-det --max-set-size 512 [--seed SEED]
    python scripts/run.py amazon --model SubmodInt --dataset gear [--layers L]
    python scripts/run.py dspn --model Hybrid [--fixed] --data DOWNLOAD_DIR --features FEATURE_DIR [--seed SEED]

--list prints the models and datasets of an experiment; --dry-run prints the commands without running them. Results
are written under --out: synthetic/DATASET/MODEL/seedSEED/results.toml (test RMSE and MAE; DATASET is log-det-maxN
with --max-set-size N, whose settings are in configs/synthetic-set-size.toml), amazon/CATEGORY/MODEL.json (test
Jaccard and NDCG) and dspn/MODEL/seedSEED/early_stopping.json (test FL Eval and offline and online accuracy; seeds 0,
1 and 2 one after the other unless --seed is given). A finished run is not repeated; an interrupted synthetic or dspn
run resumes.
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASETS = {
    "synthetic": ["fac-loc", "g-cut", "g-cut-8", "int-2-1", "log", "log-det", "log-times-log-det", "log-times-sqrt", "sqrt"],
    "synthetic-set-size": ["128", "256", "512", "1024", "2048"],
    "amazon": ["apparel", "bath", "bedding", "diaper", "feeding", "gear", "health", "media"],
    "dspn": ["imagenet-100"],
}
EXPERIMENTS = ["synthetic", "amazon", "dspn"]
DEFAULT_SEED = {"synthetic": 0}
DSPN_SEEDS = [0, 1, 2]
TABLE_NAME = re.compile(r"(?P<model>[A-Za-z]+?)(?P<fixed>_F)?(?:-L(?P<layers>\d+))?")


def load_config(name):
    return tomllib.loads((REPO / "configs" / f"{name}.toml").read_text())


def tasks(config):
    """(model, fixed, layers, datasets or None for any) of each table of a config."""
    for name, table in config.items():
        if name != "common":
            m = TABLE_NAME.fullmatch(name)
            datasets = [key for key, value in table.items() if isinstance(value, dict)] or None
            yield name, m["model"], bool(m["fixed"]), m["layers"] and int(m["layers"]), datasets


def resolve(config, experiment, model, fixed, layers, dataset):
    """The config table of the task, and its options: [common], then the table's, then its entry for dataset."""
    matches = [t for t in tasks(config) if t[1] == model and t[2] == fixed and layers in (None, t[3])]
    if layers is None and len(matches) > 1:
        matches = [t for t in matches if t[3] == 2]
    if len(matches) != 1:
        sys.exit(f"configs/{experiment}.toml has no {model}{' with fixed transforms' if fixed else ''}"
                 f"{f' with {layers} layers' if layers else ''}; see --list")
    name, *_, datasets = matches[0]
    if dataset not in (datasets or DATASETS[experiment]):
        sys.exit(f"{name} has settings for {', '.join(datasets or DATASETS[experiment])}, not {dataset}")
    table = config[name]
    options = config.get("common", {}) | {k: v for k, v in table.items() if not isinstance(v, dict)}
    return name, options | table.get(dataset, {})


def flags(options):
    """--key value for each option; key = true gives --key, and a list gives --key followed by its items."""
    args = []
    for key, value in options.items():
        if value is True:
            args.append(f"--{key}")
        elif value is not False:
            args += [f"--{key}", *map(str, value if isinstance(value, list) else [value])]
    return args


def synthetic(cli, name, options):
    dataset_dir = f"{cli.dataset}-max{cli.max_set_size}" if cli.max_set_size else cli.dataset
    run_dir = cli.out / "synthetic" / dataset_dir / name / f"seed{cli.seed}"
    if (run_dir / "results.toml").exists():
        print(f"{run_dir} is finished")
        return []
    if cli.max_set_size:
        options["max-set-size"] = cli.max_set_size
    cmd = [sys.executable, "train_inference/train_test.py", options.pop("model"), cli.dataset, *flags(options),
           "--seed", str(cli.seed), "--run-dir", str(run_dir), "--log-dir", str(run_dir / "tensorboard")]
    cmd += ["--device", cli.device] if cli.device else []
    cmd += ["--compile"] if cli.compile else []
    cmd += ["--resume-training"] if (run_dir / "model_latest.pt").exists() else []
    return [(REPO, cmd)]


def amazon(cli, name, options):
    results = cli.out / "amazon"
    if (results / cli.dataset / f"{name}.json").exists():
        print(f"{results / cli.dataset / name}.json is finished")
        return []
    program = [sys.executable, "-m", "subset_select.run_subset_selection_real_data", "--phi_concave_func", cli.dataset,
               "--machine", "USER_MACHINE_REAL", *(["--device", cli.device] if cli.device else [])]
    steps = []
    if not (REPO / f"Data_Real/real_dataset/Summarization_dataset__{cli.dataset}.pickle").exists():
        steps.append((REPO, [*program, "--method", "DeepSet", "--only_dataset_generation"]))
    steps.append((REPO, [*program, *flags(options), "--load_real_dataset", "--run_name", name,
                         "--results_dir", str(results)]))
    return steps


def dspn(cli, name, options):
    if not (cli.data and cli.features):
        sys.exit("dspn needs --data (ImageNet-100 download directory) and --features (CLIP feature directory)")
    data, root, dset = cli.data.resolve(), cli.features.resolve(), "IN100"
    val_root = root.with_name(root.name + "_val")
    steps = []
    if not (root / dset / "dataset/Y_train.npy").exists():
        steps.append(["download.py", "--data", data])
        steps.append(["extract_clip.py", "--data", data, "--root", root, "--dset", dset])
    if not (val_root / dset / "dataset/Y_train.npy").exists():
        steps.append(["make_val_split.py", "--src_root", root, "--dst_root", val_root, "--dset", dset])
    if not (val_root / dset / "passive_idx/passive_samples.pkl").exists():
        steps.append(["make_passive_pairs.py", "--root", val_root, "--dset", dset])
    if not (val_root / dset / "feedback_idx/fl_feedback_summary.npy").exists():
        steps.append(["make_feedback_summaries.py", "--root", val_root, "--dset", dset, "--kw", "0.3"])
    extra = options.pop("extra", "")
    for seed in DSPN_SEEDS if cli.seed is None else [cli.seed]:
        steps.append(["run_long.py", *flags(options), "--extra", f"{extra} --seed {seed}".strip(), "--runs_dir",
                      cli.out / "dspn" / name, "--root", root, "--val_root", val_root, "--dset", dset, "--name", f"seed{seed}"])
    return [(REPO / "dspn", [sys.executable, *map(str, step)]) for step in steps]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=EXPERIMENTS)
    ap.add_argument("--model", help="as in the config, without _F and -L<layers>, e.g. SubmodInt")
    ap.add_argument("--dataset", help="a synthetic dataset or an Amazon category; see --list")
    ap.add_argument("--max-set-size", type=int, help="synthetic on log-det: the largest set, 128, 256, 512, 1024 or 2048")
    ap.add_argument("--layers", type=int, help="number of layers (default: 2 where the config has several)")
    ap.add_argument("--fixed", action="store_true", help="SubmodInt with fixed transforms (SubmodInt_F)")
    ap.add_argument("--seed", type=int, help="default 0 (dspn: 0, 1 and 2, one after the other)")
    ap.add_argument("--out", type=Path, default=Path("runs"), help="output directory (default: runs)")
    ap.add_argument("--device", help="synthetic and amazon: e.g. cpu or cuda:0 (default: the program's)")
    ap.add_argument("--compile", action="store_true", help="synthetic: compile the model (faster on a GPU)")
    ap.add_argument("--data", type=Path, help="dspn: directory for the ImageNet-100 download")
    ap.add_argument("--features", type=Path, help="dspn: directory for its CLIP features")
    ap.add_argument("--list", action="store_true", help="print the models and datasets of the experiment")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    cli = ap.parse_args()

    if cli.max_set_size and (cli.experiment != "synthetic" or cli.dataset != "log-det"):
        ap.error("--max-set-size is for synthetic --dataset log-det")
    if cli.list:
        for config_name in [cli.experiment] + (["synthetic-set-size"] if cli.experiment == "synthetic" else []):
            if config_name == "synthetic-set-size":
                print("--dataset log-det --max-set-size N:")
            for name, model, fixed, layers, datasets in tasks(load_config(config_name)):
                options = " ".join(filter(None, [f"--model {model}", fixed and "--fixed", layers and f"--layers {layers}"]))
                print(f"{options:<36} ({name}): {' '.join(datasets or DATASETS[config_name])}")
        return
    config_name = "synthetic-set-size" if cli.max_set_size else cli.experiment
    config = load_config(config_name)
    if not cli.model:
        ap.error("--model is required (see --list)")
    if cli.experiment == "dspn":
        cli.dataset = cli.dataset or DATASETS["dspn"][0]
    elif not cli.dataset:
        ap.error("--dataset is required (see --list)")
    if cli.experiment == "amazon" and cli.seed is not None:
        ap.error("the Amazon pipeline has a fixed seed")
    if cli.experiment == "dspn" and cli.device:
        ap.error("dspn trains on the GPU")
    if cli.compile and cli.experiment != "synthetic":
        ap.error("--compile is for synthetic")
    cli.seed = DEFAULT_SEED.get(cli.experiment) if cli.seed is None else cli.seed
    cli.out = cli.out.resolve()

    key = str(cli.max_set_size) if cli.max_set_size else cli.dataset
    name, options = resolve(config, config_name, cli.model, cli.fixed, cli.layers, key)
    steps = {"synthetic": synthetic, "amazon": amazon, "dspn": dspn}[cli.experiment](cli, name, options)
    env = os.environ | {"PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")]))}
    for cwd, cmd in steps:
        print(f"(cd {shlex.quote(str(cwd))} && {shlex.join(cmd)})", flush=True)
        if not cli.dry_run:
            subprocess.run(cmd, cwd=cwd, env=env, check=True)


if __name__ == "__main__":
    main()
