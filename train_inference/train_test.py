import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Optional, override

import clap
import torch
import torch.nn as nn
import torch.nn.functional as F
from clap import arg, long
from plum import dispatch
from serde import ClassDeserializer, ClassSerializer, serde
from serde.toml import from_toml, to_toml
from torch import Tensor
from torch.optim import Adam, AdamW, Muon, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.combinatorial_dataset import create_combinatorial_dataset
from data.interaction_dataset import InteractionDataset
from data.synthetic_data import SyntheticSetDataset
from models.baselines import (
    DSF,
    BaselineConfig,
    DeepSet,
    SetTransformer,
    subnet_flex_dual,
)
from models.edsf.dsf import ExtendedDSF
from models.submodint import SubmodInt, SubmodIntConfig


class Model(StrEnum):
    SubmodInt = auto()
    """SubmodInt."""
    DS = auto()
    """DeepSet."""
    ST = auto()
    """Set Transformer."""
    DSF = auto()
    """DSF."""
    EDSF = auto()
    """Extended DSF."""
    FlexSubNet = auto()
    """FlexSubNet."""


class SynData(StrEnum):
    Int2_1 = auto()
    """2-ary (pairwise) interaction, 1-layer INT dataset."""
    Int2_2 = auto()
    """2-ary (pairwise) interaction, 2-layer INT dataset."""
    Int3_1 = auto()
    """Not implemented."""
    Int3_2 = auto()
    """Not implemented."""
    Int4_1 = auto()
    """Not implemented."""
    Int4_2 = auto()
    """Not implemented."""
    FacLoc = auto()
    """Facility location."""
    Log = auto()
    LogDet = auto()
    LogTimesLogDet = auto()
    GCut = auto()
    GCut8 = auto()
    Sqrt = auto()
    LogTimesSqrt = auto()


class ModelOptimizer(StrEnum):
    Adam = auto()
    AdamW = auto()
    Muon = auto()


class IntGenerator(StrEnum):
    Paper = auto()
    """The generator behind the paper's INT results (data/synthetic_data.py)."""
    RandomMlp = auto()
    """Random positive-MLP interactions (data/interaction_dataset.py)."""


class DeviceSerializer(ClassSerializer):
    @dispatch
    def serialize(self, value: torch.device) -> str:
        return str(value)


class DeviceDeserializer(ClassDeserializer):
    @override
    @dispatch
    def deserialize(self, cls: type[torch.device], value: str) -> torch.device:
        return torch.device(value)


@serde
@clap.group(title="Dataset options")
class DatasetOpts:
    feature_dimension: int = arg(long, default_value=10, value_name="D")
    """The feature dimension."""
    gamma: float = arg(long, default_value=2.5)
    """The constant gamma for the interaction dataset."""
    max_set_size: int = arg(long, default_value=32, value_name="S")
    """The maximum set size."""
    num_samples: int = arg(long, default_value=10_000, value_name="N")
    """The number of samples to generate."""
    ground_set_size: int = arg(long, default_value=100, value_name="N")
    """The number of elements in the ground set. This is not used in INT
    datasets, where the ground set is R^d."""
    int_generator: IntGenerator = arg(long, default_value=IntGenerator.Paper)
    """The generator used for the INT datasets."""


@serde
@clap.group(title="Training options")
class TrainOpts:
    num_epochs: int = arg(long, default_value=150, value_name="N_EPOCHS")
    """The maximum number of epochs to train for."""
    learning_rate: float = arg(long="lr", default_value=1e-3, value_name="LR")
    """The learning rate."""
    optimizer: ModelOptimizer = arg(long, default_value=ModelOptimizer.Adam)
    """The optimizer to use."""
    weight_decay: float = arg(long, default_value=1e-4, value_name="DECAY")
    """Weight decay."""
    early_stopping_patience: int = arg(long, default_value=30, value_name="N")
    """The early stopping patience."""
    lr_decay_factor: float = arg(long, default_value=0.1, value_name="F")
    """Factor by which ReduceLROnPlateau multiplies the learning rate."""
    lr_patience: int = arg(long, default_value=10, value_name="N")
    """Epochs without validation improvement before ReduceLROnPlateau decays the learning rate."""
    resume_training: bool = arg(long)
    """Resume training from the latest checkpoint."""


@serde(class_serializer=DeviceSerializer(), class_deserializer=DeviceDeserializer())
@clap.command(max_term_width=100)
class Cli(clap.Parser):
    """Train and evaluate models on synthetic datasets."""

    model: Model = arg()
    """The model to train and evaluate.

    To configure SubmodInt, see the SubmodInt config section of this help message.
    To configure the baselines, see the Baseline config section of this help message.
    """
    dataset: SynData = arg()
    """The dataset to use."""

    seed: int = arg(long)
    """The seed for all random number generators."""
    run_dir: Path = arg(long, value_name="DIR")
    """The path to the directory where results, checkpoints, and config for this
    run are saved. Existing files are overwritten."""
    log_dir: Path = arg(long, value_name="DIR")
    """The directory where TensorBoard logs will be saved."""
    device: torch.device = arg(
        long, default_value=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    """The device to use for training and evaluation."""
    batch_size: int = arg(long, default_value=128, value_name="B")
    """The batch size."""
    eval_best_model: bool = arg(long)
    """Evaluate the best checkpoint in the given run directory."""
    compile: bool = arg(long)
    """Compile the model.

    This is recommended since it gives a significant speedup.
    """

    dataset_opts: DatasetOpts = DatasetOpts()
    submodint_opts: SubmodIntConfig = SubmodIntConfig()
    baseline_opts: BaselineConfig = BaselineConfig()
    train_opts: TrainOpts = TrainOpts()


@dataclass(slots=True)
class Checkpoint:
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any]
    train_generator_state: Tensor
    val_generator_state: Tensor
    epoch: int


torch.serialization.add_safe_globals([
    Adam,
    Checkpoint,
    defaultdict,
    dict,
    getattr,
    torch.Generator,
])


@serde
class Result:
    @serde
    class Value:
        mean: float
        std_error: float

    rmse: Value
    mae: Value


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    mse, mae, n = 0.0, 0.0, 0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        if isinstance(model, subnet_flex_dual):
            y_pred, *_ = model(X)
            y_pred = y_pred.squeeze(-1)
        elif isinstance(model, (SubmodInt, ExtendedDSF)):
            y_pred = model(X).squeeze(-1)
        else:
            y_pred = model(X, torch.zeros(X.shape[0], device=device)).squeeze(-1)
        diff = y_pred - y
        mse += (diff**2).sum().item()
        mae += diff.abs().sum().item()
        n += y.numel()
    return math.sqrt(mse / n), mae / n


@torch.inference_mode()
def get_mean_and_std_error(model: nn.Module, loader: DataLoader, device: torch.device) -> Result:
    model.eval()
    rmses = []
    maes = []
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        if isinstance(model, subnet_flex_dual):
            y_pred, *_ = model(X)
            y_pred = y_pred.squeeze(-1)
        elif isinstance(model, (SubmodInt, ExtendedDSF)):
            y_pred = model(X).squeeze(-1)
        else:
            y_pred = model(X, torch.zeros(X.shape[0], device=device)).squeeze(-1)
        rmses.append(math.sqrt(F.mse_loss(y_pred, y).item()))
        maes.append(F.l1_loss(y_pred, y).item())
    rmses = torch.tensor(rmses)
    maes = torch.tensor(maes)
    return Result(
        rmse=Result.Value(
            torch.mean(rmses).item(), torch.std(rmses).item() / (len(loader) ** 0.5)
        ),
        mae=Result.Value(torch.mean(maes).item(), torch.std(maes).item() / (len(loader) ** 0.5)),
    )


def create_datasets(
    dataset_name: SynData, device: torch.device, opts: DatasetOpts
) -> tuple[Dataset, Dataset, Dataset]:
    train_set: Dataset
    val_set: Dataset
    test_set: Dataset

    match dataset_name:
        case (
            SynData.Log
            | SynData.FacLoc
            | SynData.LogDet
            | SynData.GCut
            | SynData.GCut8
            | SynData.Sqrt
            | SynData.LogTimesSqrt
            | SynData.LogTimesLogDet
        ):
            ds: str
            match dataset_name:
                case SynData.Log: ds = "torch_log"
                case SynData.FacLoc: ds = "fac_loc"
                case SynData.LogDet: ds = "DPP"
                case SynData.LogTimesLogDet: ds = "LOG_DPP"
                case SynData.GCut: ds = "Gcut"
                case SynData.GCut8: ds = "Gcut0.8"
                case SynData.Sqrt: ds = "torch_sqrt"
                case SynData.LogTimesSqrt: ds = "LOG_SQRT"
            train_set, val_set, test_set = create_combinatorial_dataset(
                str(device),
                phi_concave_func=ds,
                num_features=opts.feature_dimension,
                max_set_size=opts.max_set_size,
                num_samples=opts.num_samples,
                total_number_of_elements_in_ground_set=opts.ground_set_size,
                folds=(0.7, 0.15, 0.15),
            )
        case (
            SynData.Int2_1
            | SynData.Int2_2
            | SynData.Int2_2
            | SynData.Int3_1
            | SynData.Int3_2
            | SynData.Int4_1
            | SynData.Int4_2
        ):
            interaction_arity, num_layers_syn_data = (
                int(s) for s in (str(dataset_name)[3:].split("_"))
            )
            match opts.int_generator:
                case IntGenerator.Paper:
                    if interaction_arity != 2:
                        msg = "The paper INT generator only has pairwise interactions."
                        raise NotImplementedError(msg)
                    dataset = SyntheticSetDataset(
                        N=opts.num_samples,
                        d=opts.feature_dimension,
                        k=opts.max_set_size,
                        gamma=opts.gamma,
                        num_layers=num_layers_syn_data,
                    )
                case IntGenerator.RandomMlp:
                    dataset = InteractionDataset(
                        N=opts.num_samples,
                        d=opts.feature_dimension,
                        k=opts.max_set_size,
                        gamma=opts.gamma,
                        interaction_arity=interaction_arity,
                        num_layers=num_layers_syn_data,
                    )
            N_train = int(0.70 * opts.num_samples)
            N_val = int(0.15 * opts.num_samples)
            N_test = opts.num_samples - N_train - N_val
            train_set, val_set, test_set = random_split(dataset, [N_train, N_val, N_test])

    return train_set, val_set, test_set


def create_dataloaders(
    train_set: Dataset,
    val_set: Dataset,
    test_set: Dataset,
    batch_size: int,
    train_generator: torch.Generator,
    val_generator: torch.Generator,
    test_generator: torch.Generator,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    return (
        DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=1,
            pin_memory=True,
            generator=train_generator,
        ),
        DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=1,
            pin_memory=True,
            generator=val_generator,
        ),
        DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=1,
            pin_memory=True,
            generator=test_generator,
        ),
    )


def create_model_and_optimizer(
    model_name: Model,
    feature_dim: int,
    gamma: float,
    max_set_size: int,
    submodint_config: SubmodIntConfig,
    baseline_config: BaselineConfig,
    optimizer_name: ModelOptimizer,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[nn.Module, Optimizer]:
    model: nn.Module
    match model_name:
        case Model.SubmodInt:
            model = SubmodInt.from_config(
                feature_dim=feature_dim,
                gamma=gamma,
                max_set_size=max_set_size,
                config=submodint_config,
            ).to(device)
        case Model.DS:
            model = DeepSet(
                num_feature=feature_dim,
                in_d=baseline_config.in_d,
                hidden_layers=baseline_config.hidden_layers,
                dev=str(device),
            ).to(device)
        case Model.ST:
            model = SetTransformer(
                num_feature=feature_dim,
                in_d=1,
                hidden_layers=baseline_config.hidden_layers,
            ).to(device)
        case Model.DSF:
            model = DSF(
                input_feature_dim=feature_dim,
                hidden_layers=baseline_config.hidden_layers,
                num_steps=baseline_config.num_rnn_steps,
                readout=baseline_config.dsf_readout,
            ).to(device)
        case Model.EDSF:
            model = ExtendedDSF(
                in_dim=feature_dim,
                hidden_sizes=baseline_config.hidden_layers,
                alpha=100.0,
            ).to(device)
        case Model.FlexSubNet:
            model = subnet_flex_dual(
                num_feature=feature_dim,
                in_d=baseline_config.in_d,
                hidden_layers=baseline_config.hidden_layers,
                num_rnn_steps=baseline_config.num_rnn_steps,
                dev=str(device),
            ).to(device)
    match optimizer_name:
        case ModelOptimizer.Adam:
            return model, Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        case ModelOptimizer.AdamW:
            return model, AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        case ModelOptimizer.Muon:
            return model, Muon(model.parameters(), lr=lr, weight_decay=weight_decay)


def load_run(run_dir: Path, device: torch.device = torch.device("cpu")) -> tuple[Cli, nn.Module]:
    """The configuration of a finished run in run_dir and its model, with the weights of its best checkpoint."""
    config = (run_dir / "config.toml").read_text().split("\n[model_info]")[0]
    cli = from_toml(Cli, config)
    model, _ = create_model_and_optimizer(
        cli.model,
        cli.dataset_opts.feature_dimension,
        cli.dataset_opts.gamma,
        cli.dataset_opts.max_set_size,
        cli.submodint_opts,
        cli.baseline_opts,
        cli.train_opts.optimizer,
        cli.train_opts.learning_rate,
        cli.train_opts.weight_decay,
        device,
    )
    sys.modules["__main__"].__dict__.setdefault("Checkpoint", Checkpoint)
    checkpoint: Checkpoint = torch.load(run_dir / "model_best.pt", weights_only=False, map_location=device)
    model.load_state_dict(checkpoint.model_state)
    return cli, model


def train_model(
    model: nn.Module,
    opt: Optimizer,
    scheduler: ReduceLROnPlateau,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    early_stopping_patience: int,
    train_generator: torch.Generator,
    val_generator: torch.Generator,
    latest_checkpoint_path: Path,
    resume_epoch: Optional[int],
    resume_best: Optional[tuple[Checkpoint, float]],
    device: torch.device,
    tb_writer: SummaryWriter,
) -> Optional[Checkpoint]:
    best_val = float("inf")
    best_checkpoint: Optional[Checkpoint] = None
    epochs_no_improve: int = 0
    if resume_best is not None and resume_epoch is not None:
        best_checkpoint, best_val = resume_best
        epochs_no_improve = resume_epoch - best_checkpoint.epoch
        if epochs_no_improve >= early_stopping_patience:
            return best_checkpoint
    loss_fn = nn.MSELoss()

    for epoch in tqdm(range((resume_epoch or 0) + 1, num_epochs + 1)):
        model.train()
        running_mse = 0.0
        running_mae = 0.0
        count = 0
        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            if isinstance(model, (SubmodInt, ExtendedDSF)):
                y_pred = model(X).squeeze(-1)
                loss = loss_fn(y_pred, y)
                loss.backward()
            elif isinstance(model, subnet_flex_dual):
                y_pred, x_input_to_first_integrand, outer_integral_second_out = model(X)
                y_pred = y_pred.squeeze(-1)
                loss = torch.sum((y_pred - y) ** 2) + torch.sum(
                    (x_input_to_first_integrand - outer_integral_second_out) ** 2
                )
                loss.backward()
            else:
                y_pred = model(X, torch.zeros(X.shape[0], device=device)).squeeze(-1)
                loss = loss_fn(y_pred, y)
                loss.backward()
            opt.step()

            running_mse += torch.sum((y_pred - y) ** 2).item()
            running_mae += F.l1_loss(y_pred.squeeze(-1), y).item() * y.numel()
            count += y.numel()

        val_rmse, val_mae = evaluate(model, val_loader, device)
        scheduler.step(val_rmse)

        tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)
        tb_writer.add_scalar("RMSE/train", math.sqrt(running_mse / max(count, 1)), epoch)
        tb_writer.add_scalar("RMSE/val", val_rmse, epoch)
        tb_writer.add_scalar("MAE/train", running_mae / max(count, 1), epoch)
        tb_writer.add_scalar("MAE/val", val_mae, epoch)

        checkpoint = Checkpoint(
            {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            opt.state_dict(),
            scheduler.state_dict(),
            train_generator.get_state(),
            val_generator.get_state(),
            epoch,
        )
        torch.save(checkpoint, latest_checkpoint_path)
        if val_rmse + 1e-12 < best_val:
            best_val = val_rmse
            best_checkpoint = checkpoint
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                tb_writer.add_text(
                    "Early stopping",
                    f"Early stopping triggered after {epochs_no_improve} epochs",
                    epoch,
                )
                break

    return best_checkpoint


def main():
    cli = Cli.parse()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    cli.run_dir.mkdir(exist_ok=True, parents=True)
    run_config_file = cli.run_dir / "config.toml"
    run_config_file.write_text(to_toml(cli))

    best_checkpoint: Optional[Checkpoint] = None
    best_checkpoint_path = cli.run_dir / "model_best.pt"
    latest_checkpoint_path = cli.run_dir / "model_latest.pt"
    if cli.eval_best_model:
        if not best_checkpoint_path.exists():
            msg = f"Checkpoint file does not exist at {best_checkpoint_path}."
            raise RuntimeError(msg)
        best_checkpoint = torch.load(best_checkpoint_path)
    if cli.train_opts.resume_training:
        if not latest_checkpoint_path.exists():
            msg = f"Checkpoint file does not exist at {latest_checkpoint_path}."
            raise RuntimeError(msg)
        best_checkpoint = torch.load(latest_checkpoint_path)

    torch.manual_seed(42)
    train_set, val_set, test_set = create_datasets(cli.dataset, cli.device, cli.dataset_opts)

    torch.manual_seed(cli.seed)
    train_generator = torch.Generator().manual_seed(int(torch.randint(0, 2**63 - 1, (1,)).item()))
    val_generator = torch.Generator().manual_seed(int(torch.randint(0, 2**63 - 1, (1,)).item()))
    test_generator = torch.Generator().manual_seed(int(torch.randint(0, 2**63 - 1, (1,)).item()))
    if best_checkpoint is not None:
        train_generator.set_state(best_checkpoint.train_generator_state)
        val_generator.set_state(best_checkpoint.val_generator_state)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_set,
        val_set,
        test_set,
        cli.batch_size,
        train_generator,
        val_generator,
        test_generator,
    )
    model, opt = create_model_and_optimizer(
        cli.model,
        cli.dataset_opts.feature_dimension,
        cli.dataset_opts.gamma,
        cli.dataset_opts.max_set_size,
        cli.submodint_opts,
        cli.baseline_opts,
        cli.train_opts.optimizer,
        cli.train_opts.learning_rate,
        cli.train_opts.weight_decay,
        cli.device,
    )
    num_model_params = sum(param.numel() for param in model.parameters())
    scheduler = ReduceLROnPlateau(
        opt, factor=cli.train_opts.lr_decay_factor, patience=cli.train_opts.lr_patience
    )

    with open(run_config_file, "a") as f:
        f.write("\n[model_info]")
        f.write(f"\nnum_params = {num_model_params}\n")

    if cli.compile:
        model.compile()

    resume_epoch: Optional[int] = None
    if best_checkpoint is not None:
        model.load_state_dict(best_checkpoint.model_state)
        opt.load_state_dict(best_checkpoint.optimizer_state)
        scheduler.load_state_dict(best_checkpoint.scheduler_state)
        resume_epoch = best_checkpoint.epoch

    resume_best: Optional[tuple[Checkpoint, float]] = None
    if cli.train_opts.resume_training and best_checkpoint_path.exists():
        prior_best = torch.load(best_checkpoint_path)
        val_generator_state = val_generator.get_state()
        model.load_state_dict(prior_best.model_state)
        prior_best_val, _ = evaluate(model, val_loader, cli.device)
        val_generator.set_state(val_generator_state)
        model.load_state_dict(best_checkpoint.model_state)
        resume_best = (prior_best, prior_best_val)

    cli.log_dir.mkdir(exist_ok=True, parents=True)
    tb_writer = SummaryWriter(cli.log_dir)

    if not cli.eval_best_model:
        best_checkpoint = train_model(
            model,
            opt,
            scheduler,
            train_loader,
            val_loader,
            cli.train_opts.num_epochs,
            cli.train_opts.early_stopping_patience,
            train_generator,
            val_generator,
            latest_checkpoint_path,
            resume_epoch,
            resume_best,
            cli.device,
            tb_writer,
        )
        torch.save(best_checkpoint, best_checkpoint_path)

    if best_checkpoint is None:
        msg = "Did not train or load model from disk."
        raise RuntimeError(msg)

    model.load_state_dict(best_checkpoint.model_state)
    result = get_mean_and_std_error(
        model,
        test_loader,
        cli.device,
    )
    tb_writer.add_hparams(
        {
            "num_params": num_model_params,
            "batch_size": cli.batch_size,
            "lr": cli.train_opts.learning_rate,
            "num_epochs": cli.train_opts.num_epochs,
            "early_stopping_patience": cli.train_opts.early_stopping_patience,
        },
        {
            "rmse_mean": result.rmse.mean,
            "rmse_std_error": result.rmse.std_error,
            "mae_mean": result.mae.mean,
            "mae_std_error": result.mae.std_error,
        },
    )
    (cli.run_dir / "results.toml").write_text(to_toml(result))


if __name__ == "__main__":
    main()
