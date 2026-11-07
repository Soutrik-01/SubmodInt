import argparse
import json
import os

import torch

from common import logger, set_log
from common.utils import configure, create_data_folders, save_into_pickle
from models.baselines import (
    DSF,
    DeepSet,
    MaskedSetTransformer,
    deep_subm_baseline,
    mixture_submodular,
    modular_func,
    neural_subm_one_level,
    subnet_flex_recurrent,
)
from models.submodint import SubmodInt
from subset_select.learn_eval_subset_select_real_data import (
    evaluate_trained_model,
    learn_neural_submodular_function_from_ArgMax,
)
from subset_select.process_real_data import load_real_dataset, real_dataset

SetFuncDict = {
    "DeepSet": DeepSet,
    "neural_subm_one_level": neural_subm_one_level,
    "modular_func": modular_func,
    "subnet_flex_recurrent": subnet_flex_recurrent,
    "deep_subm_baseline": deep_subm_baseline,
    "mixture_submodular": mixture_submodular,
    "SetTransformer": MaskedSetTransformer,
    "SubmodInt": SubmodInt,
    "DSF": DSF,
}
dataset_list = [
    "toys",
    "decor",
    "moms",
    "furniture",
    "gear",
    "trollers",
    "gifts",
    "carseats",
    "pottytrain",
    "bath",
    "health",
    "diaper",
    "bedding",
    "safety",
    "feeding",
    "apparel",
    "media",
]


def generate_dataset(av):
    if av.load_real_dataset is False:
        real_data = real_dataset(av)
        real_data.split_into_training_test()
        real_data.save()

    if av.load_real_dataset is True:
        real_data = load_real_dataset(av)

    U_train, U_validation, U_test = real_data.U_sets
    S_train, S_validation, S_test = real_data.S_sets
    av.max_len = real_data.max_len
    return U_train, U_validation, U_test, S_train, S_validation, S_test


def learn_and_eval(av, TrainingData, ValidationData, TestData):
    submodular_function = learn_neural_submodular_function_from_ArgMax(
        av, TrainingData, ValidationData, TestData
    )
    NO_FLAG = submodular_function.train()
    if NO_FLAG == 0:
        return -1, -1, -1, -1

    eval_obj = evaluate_trained_model(av)
    eval_obj.predict(ValidationData, TestData)
    jc_dev, map_dev, jc_test, map_test = eval_obj.compute_error()
    print(jc_dev, map_dev, jc_test, map_test)
    av.extra_metric_values = eval_obj.extra_metrics()
    return jc_dev, map_dev, jc_test, map_test


def main_subset_selection_command_line_per_dataset(av=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_features", type=int, default=768)
    ap.add_argument("--phi_concave_func", type=str, default="NON_")
    ap.add_argument("--machine", type=str, default="NONE")

    ap.add_argument("--in_d_umnn", type=int, default=3)
    ap.add_argument("--PERM_NETWORK_LATENT_DIM", type=int, default=10)
    ap.add_argument("--hidden_layers_umnn", type=int, default=50)
    ap.add_argument("--device", type=str, default="cuda:0")

    ap.add_argument("--method", type=str, default="NONE_")
    ap.add_argument("--load_real_dataset", action="store_true")
    ap.add_argument("--load_init_model", action="store_true")
    ap.add_argument("--load_init_perm_model", action="store_true")

    ap.add_argument("--only_dataset_generation", action="store_true")
    ap.add_argument("--num_valid_init_models", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--interaction_fn", choices=["a", "b"], default="b", help="SubmodInt: a = concat, b = dot")
    ap.add_argument("--submodint_width", type=int, default=32)
    ap.add_argument("--submodint_m1", action="store_true", help="SubmodInt: keep the M1 modular term")
    ap.add_argument("--submodint_layers", type=int, default=2)
    ap.add_argument("--submodint_interaction_dims", type=int, nargs="+", default=[10, 10])
    ap.add_argument("--submodint_independent", action="store_true", help="SubmodInt: separate parameters per layer")
    ap.add_argument("--submodint_raw_modular", action="store_true", help="SubmodInt: add a modular term on raw features")
    ap.add_argument("--submodint_modular_lr", type=float, default=0.0, help="learning rate of that term; 0 = --learning_rate")
    ap.add_argument("--submodint_freeze_modular", action="store_true", help="SubmodInt: do not train that term")
    ap.add_argument("--submodint_count_init", type=float, default=0.0,
                    help="SubmodInt: start that term at gamma * log(1 + training count) with this gamma; 0 = random")
    ap.add_argument("--submodint_gate", action="store_true",
                    help="SubmodInt: scale the projected set model by a learnable gate >= 0 that starts at 0")
    ap.add_argument("--submodint_gate_lr", type=float, default=0.0, help="learning rate of the gate; 0 = --learning_rate")
    ap.add_argument("--run_name", type=str, default="", help="Distinguishes output files of configs")
    ap.add_argument("--results_dir", type=str, default="results_amazon")

    av = ap.parse_args()
    if (av.submodint_count_init or av.submodint_freeze_modular) and not av.submodint_raw_modular:
        ap.error("--submodint_count_init and --submodint_freeze_modular need --submodint_raw_modular")

    av.seed = 760671

    av.Tag = "Summary_seed_"
    create_data_folders(av.machine)

    av.hidden_layers_umnn = [av.hidden_layers_umnn]
    av.Notebook = False
    av.folds = [0.33, 0.33, 0.33]

    av.nb_steps = 100

    av.error_mode = "summarization"
    av.if_save = True
    av.if_intermed_save = True
    av.MAX_ITER = 100
    av.pad = False
    av.sinkhorn = True
    av.submodular_function = SetFuncDict[av.method]

    av.set_func_str = (
        av.method
        + "_init_condition_check_LR_"
        + str(av.learning_rate)
        + "weight_decay_"
        + str(av.weight_decay)
        + "hidden"
        + str(av.hidden_layers_umnn[0])
        + str(av.folds[0])
        + "_"
        + str(av.folds[1])
        + "_"
        + str(av.folds[2])
    )
    if av.run_name:
        av.set_func_str += "_" + av.run_name

    av.generated_data_filename = "Summarization_dataset_"
    U_train, U_validation, U_test, S_train, S_validation, S_test = generate_dataset(av)
    print(av)
    if av.only_dataset_generation == True:
        return

    TrainingData, ValidationData, TestData = {}, {}, {}
    TrainingData["U_train"] = U_train
    TrainingData["S_train"] = S_train
    TestData["U_test"] = U_test
    TestData["S_test"] = S_test
    ValidationData["U_dev"] = U_validation
    ValidationData["S_dev"] = S_validation

    metrics = {"jc_dev": [], "map_dev": [], "jc_test": [], "map_test": []}
    iter = 1
    for ITER_ in range(av.MAX_ITER):
        torch.cuda.empty_cache()

        av.task = av.Tag + str(av.seed) + "_" + str(iter)

        av.logpath = (
            "logDir_summarize"
            + av.method
            + "/logfile"
            + av.task
            + "_"
            + av.set_func_str
            + "_"
            + av.phi_concave_func
        )
        print(av)

        jc_dev, map_dev, jc_test, map_test = learn_and_eval(
            av, TrainingData, ValidationData, TestData
        )

        if jc_dev != -1:
            metrics["jc_dev"].append(jc_dev)
            metrics["map_dev"].append(map_dev)
            metrics["jc_test"].append(jc_test)
            metrics["map_test"].append(map_test)
            for k, v in av.extra_metric_values.items():
                metrics.setdefault(k, []).append(v)
            iter = iter + 1
            if iter == av.num_valid_init_models + 1:
                break
    mean_values = {}
    for msr in list(metrics.keys()):
        mean_values[msr] = sum(metrics[msr]) / len(metrics[msr])

    save_error_file = (
        configure(av.machine)["score_path"]
        + av.task
        + "_"
        + av.phi_concave_func
        + "_"
        + av.set_func_str
        + "_score_file"
    )

    av.logpath = (
        "logDir_summarize"
        + av.method
        + "/logfile"
        + av.task
        + "_"
        + av.set_func_str
        + "_"
        + av.phi_concave_func
        + "_score_file"
    )
    set_log(av)
    for iter in range(av.num_valid_init_models):
        logger.info(
            "JCDev, MAPDev, JCTest, MAPTest at iteration: %d: %0.4f, %0.4f AND %0.4f, %0.4f",
            iter,
            metrics["jc_dev"][iter],
            metrics["map_dev"][iter],
            metrics["jc_test"][iter],
            metrics["map_test"][iter],
        )

    logger.info(
        "Means: %0.4f, %0.4f AND %0.4f, %0.4f ",
        mean_values["jc_dev"],
        mean_values["map_dev"],
        mean_values["jc_test"],
        mean_values["map_test"],
    )
    save_into_pickle(metrics, save_error_file)

    results_file = os.path.join(av.results_dir, av.phi_concave_func, (av.run_name or av.method) + ".json")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(
            {
                "method": av.method,
                "category": av.phi_concave_func,
                "num_params": av.num_params,
                "config": {k: getattr(av, k) for k in ("learning_rate", "weight_decay", "epochs", "in_d_umnn",
                                                       "hidden_layers_umnn", "interaction_fn", "submodint_width",
                                                       "submodint_m1", "submodint_layers", "submodint_interaction_dims",
                                                       "submodint_independent", "submodint_raw_modular",
                                                       "submodint_modular_lr", "submodint_freeze_modular",
                                                       "submodint_count_init", "submodint_gate",
                                                       "submodint_gate_lr")},
                "attempts": ITER_ + 1,
                "runs": metrics,
                "mean": mean_values,
            },
            f,
            indent=1,
        )


if __name__ == "__main__":
    main_subset_selection_command_line_per_dataset()
