import copy
import math
import os
import time

import torch
from sklearn.metrics import ndcg_score

from common import logger, set_log
from common.utils import *
from models.baselines import (
    DSF,
    MaskedSetTransformer,
    ProjectedSetModel,
    neural_subm_from_argmax,
    set_model_from_argmax,
)
from models.submodint import InteractionFn, SubmodInt, SubmodIntConfig
from subset_select.permutation_generator import *


def greedy_picks(U_list, budgets, model):
    """Greedy picks per registry; registries sharing an item pool share one greedy run,
    since the greedy sequence does not depend on the budget."""
    groups = []
    for i, (U, budget) in enumerate(zip(U_list, budgets)):
        for group in groups:
            if group[0] is U or (group[0].shape == U.shape and torch.equal(group[0], U)):
                group[1] = max(group[1], budget)
                group[2].append(i)
                break
        else:
            groups.append([U, budget, [i]])
    picks = [None] * len(U_list)
    rng_state = torch.get_rng_state()
    for U, budget, members in groups:
        seq = model.maximize(U.to(model.device), budget)
        for i in members:
            picks[i] = seq
    torch.set_rng_state(rng_state)
    return picks


def fit_modular_to_counts(modular, U, S_list, gamma):
    """Sets the per-item term's pre-activation w . x_v + b to gamma * log(1 + number of training sets
    containing v) by minimum-norm least squares over the pool; returns the largest fit error."""
    counts = torch.zeros(U.shape[0], dtype=torch.float64)
    for S in S_list:
        counts[S] += 1
    A = torch.cat([U.double(), torch.ones(U.shape[0], 1, dtype=torch.float64)], dim=1)
    target = gamma * torch.log1p(counts)
    theta = torch.linalg.pinv(A) @ target
    with torch.no_grad():
        modular.weight.copy_(theta[:-1].unsqueeze(0))
        modular.bias.copy_(theta[-1:])
        return (modular(U.float()).squeeze(-1).double() - target).abs().max().item()


def old_notebook_ndcg(picks10, T):
    """NDCG as computed by old-main's Subset_select_table.ipynb (behind the paper's table)."""
    candidates = list(set(picks10) | set(T))
    labels = [int(c in T) for c in candidates]
    scores = [1 / (picks10.index(c) + 1) if c in picks10 else 0 for c in candidates]
    return ndcg_score([labels], [scores])


def flexsubnet_precision(picks, T):
    """FlexSubNet's second metric: compute_precision_acc over the union of the |T| picks and T."""
    candidates = union(picks, T)
    labels = [1 if c in T else 0 for c in candidates]
    predictions = [1 / (picks.index(c) + 1) if c in picks else 0 for c in candidates]
    return compute_precision_acc(labels, predictions)


def evaluate_all_metrics(U_list, ArgMax_list, model):
    """FlexSubNet's Jaccard and precision of the first |T| greedy picks, NDCG@10 of the first 10,
    the paper's Jaccard and NDCG of the first 10 picks, and their standard errors."""
    targets = [T.tolist() for T in ArgMax_list]
    picks = greedy_picks(U_list, [max(len(T), 10) for T in targets], model)
    per = {"mjc": [], "map": [], "ndcg10": [], "jc10_old": [], "ndcg_old": []}
    for seq, T in zip(picks, targets):
        per["mjc"].append(jaccard_similarity(seq[: len(T)], T))
        per["map"].append(flexsubnet_precision(seq[: len(T)], T))
        per["ndcg10"].append(ndcg_at_k(seq, T, 10))
        per["jc10_old"].append(jaccard_similarity(seq[:10], T))
        per["ndcg_old"].append(old_notebook_ndcg(seq[:10], T))
    if isinstance(model, neural_subm_from_argmax):
        for U, T in zip(U_list, targets):
            for _ in range(len(T)):
                generate_h(U, model.in_d - 1)
    out = {}
    for k, v in per.items():
        mean = sum(v) / len(v)
        out[k] = mean
        out[f"{k}_se"] = math.sqrt(sum((x - mean) ** 2 for x in v) / len(v)) / math.sqrt(len(v))
    return out


def evaluate_standalone_from_argmax(U_list, ArgMax_list, model):
    """FlexSubNet's mean Jaccard and precision of the first |T| greedy picks."""
    metrics = evaluate_all_metrics(U_list, ArgMax_list, model)
    return metrics["mjc"], metrics["map"]


class learn_neural_submodular_function_from_ArgMax:
    def __init__(self, av, TrainingData, ValidationData, TestData):
        MakeCopy(av, self)

        self.Universe_set_train = TrainingData["U_train"]
        self.ArgMax_Indx_train = TrainingData["S_train"]

        self.Universe_set_dev = ValidationData["U_dev"]
        self.ArgMax_Indx_dev = ValidationData["S_dev"]

        self.Universe_set_test = TestData["U_test"]
        self.ArgMax_Indx_test = TestData["S_test"]

        self.av = av
        self.nsf = av.submodular_function
        set_log(av)

    def build_model(self):
        if self.nsf is MaskedSetTransformer:
            net = MaskedSetTransformer(self.num_features, self.hidden_layers_umnn)
            return set_model_from_argmax(net, dev=self.device)
        if self.nsf is SubmodInt:
            config = SubmodIntConfig(
                num_layers=self.submodint_layers,
                transform_hidden_dims=[self.submodint_width],
                interaction_hidden_dims=list(self.submodint_interaction_dims),
                modular_hidden_dims=[self.submodint_width],
                independent_params=self.submodint_independent,
                fix_continuous_transforms=False,
                interaction_fn=InteractionFn(self.interaction_fn),
                no_m1=not self.submodint_m1,
                no_m2=False,
                no_phi_ccm=False,
                no_outer_transform=False,
                nb_steps=50,
            )
            net = SubmodInt.from_config(
                feature_dim=self.in_d_umnn, gamma=2.5, max_set_size=self.max_len, config=config
            )
            model = ProjectedSetModel(
                self.num_features, self.in_d_umnn, net,
                raw_modular=self.submodint_raw_modular, gate=self.submodint_gate,
            )
            if self.submodint_count_init:
                error = fit_modular_to_counts(
                    model.modular, self.Universe_set_train[0], self.ArgMax_Indx_train, self.submodint_count_init
                )
                logger.info("per-item term fit to training counts, largest error %.2e", error)
            return set_model_from_argmax(model, dev=self.device)
        if self.nsf is DSF:
            net = DSF(self.in_d_umnn, [20, 20], 2)
            return set_model_from_argmax(
                ProjectedSetModel(self.num_features, self.in_d_umnn, net), dev=self.device
            )
        return neural_subm_from_argmax(
            self.nsf,
            self.num_features,
            self.in_d_umnn,
            self.hidden_layers_umnn,
            nb_steps=self.nb_steps,
            dev=self.device,
            mode="softmax",
        )

    def train(self):
        if self.load_init_model == True:
            model_subm = self.read_init_model()
            self.init_model = copy.deepcopy(model_subm)

        else:
            model_subm = self.build_model().to(self.device)
            self.init_model = copy.deepcopy(model_subm)
        self.av.num_params = sum(p.numel() for p in model_subm.parameters())

        if self.av.sinkhorn == True:
            if self.load_init_perm_model == True:
                perm_gen = self.read_init_perm_model()
                self.init_perm_model = copy.deepcopy(perm_gen)

            else:
                perm_gen = PermutationGenerator_set(self.av).to(self.device)
                self.init_perm_model = copy.deepcopy(perm_gen)

        print(self.count_params())

        net = getattr(model_subm, "neural_net", None)
        modular, gate = getattr(net, "modular", None), getattr(net, "gate", None)
        groups = []
        if modular is not None and (self.av.submodint_modular_lr or self.av.submodint_freeze_modular):
            lr = 0.0 if self.av.submodint_freeze_modular else self.av.submodint_modular_lr
            groups.append({"params": list(modular.parameters()), "lr": lr})
        if gate is not None and self.av.submodint_gate_lr:
            groups.append({"params": [gate], "lr": self.av.submodint_gate_lr})
        if groups:
            ids = {id(p) for g in groups for p in g["params"]}
            params = [{"params": [p for p in model_subm.parameters() if id(p) not in ids]}, *groups]
        else:
            params = model_subm.parameters()
        optim_subm = torch.optim.Adam(params, self.learning_rate, weight_decay=self.weight_decay)
        if gate is not None:
            optim_subm.register_step_post_hook(lambda *_: gate.data.clamp_(min=0))
        logger.info("optimizer learning rates: %s", [g["lr"] for g in optim_subm.param_groups])

        Universe_set_train_, ArgMax_Indx_train_ = self.Universe_set_train, self.ArgMax_Indx_train

        if self.av.sinkhorn == True:
            optim_perm = torch.optim.Adam(
                perm_gen.parameters(), 1e-6, weight_decay=self.weight_decay
            )

        last_jc = 0.0

        for epoch in range(self.epochs):
            st_t = time.time()

            Universe_set_train = Universe_set_train_
            ArgMax_Indx_train = ArgMax_Indx_train_

            total_loss = 0.0

            for i in range(len(Universe_set_train)):
                U_i = Universe_set_train[i]
                S_i = ArgMax_Indx_train[i]
                Not_ArgMax_Index = find_not_in_set(
                    torch.tensor(list(range(U_i.shape[0]))), S_i
                ).to(self.device)
                U = U_i.to(self.device)
                S = S_i.to(self.device)

                if self.av.sinkhorn == False:
                    P = generate_test_permutation(S.shape[0]).to(self.device)
                    model_subm.preprocess_input(U, S, Not_ArgMax_Index)

                    loglike, out = model_subm(P)
                    loss = -loglike

                    optim_subm.zero_grad()
                    loss.backward()
                    optim_subm.step()

                if self.av.sinkhorn == True:

                    freeze(model_subm)
                    unfreeze(perm_gen)
                    P = perm_gen(U, S)

                    model_subm.preprocess_input(U, S, Not_ArgMax_Index)
                    loglike, out = model_subm(P)
                    loss1 = loglike

                    optim_perm.zero_grad()
                    loss1.backward()
                    optim_perm.step()

                    freeze(perm_gen)
                    unfreeze(model_subm)
                    Perm_mat = perm_gen(U, S)
                    P = Perm_mat

                    model_subm.preprocess_input(U, S, Not_ArgMax_Index)
                    loglike, out = model_subm(P)
                    loss = -loglike

                    optim_subm.zero_grad()
                    loss.backward()
                    optim_subm.step()

                total_loss += loss.item()

            training_time = time.time() - st_t
            logger.info("Time for this epoch %0.3f", training_time)
            logger.info("epoch %d", epoch)
            logger.info("Average Training loss: %0.4f", total_loss)
            self.trained_model = model_subm
            self.trained_perm_model = perm_gen

            jc_val, map_val = evaluate_standalone_from_argmax(
                self.Universe_set_dev, self.ArgMax_Indx_dev, model_subm
            )
            jc_test, map_test = evaluate_standalone_from_argmax(
                self.Universe_set_test, self.ArgMax_Indx_test, model_subm
            )

            validation_time = time.time() - training_time - st_t

            logger.info("Validation time %0.3f", validation_time)

            logger.info("Average val JC, MAP: %0.4f, %0.4f", jc_val, map_val)
            logger.info("Average test JC, MAP: %0.4f, %0.4f", jc_test, map_test)

            if epoch == 0 and self.load_init_model == False:
                if self.check_no_flag(total_loss) == True:
                    self.save(init=True, FLAG=False)

                else:
                    self.save(init=True, FLAG=True)
                    return 0

            self.current_epoch = epoch

            if self.av.if_intermed_save == True:
                self.save(intermediate=True)

            if jc_val > last_jc and self.av.if_save == True:
                last_jc = jc_val
                self.save()
        return 1

    def check_no_flag(self, total_loss):
        out = True
        print("checking flag")
        if total_loss > 1e10 or math.isnan(total_loss):
            out = False
            print("flagged")
        return out

    def save(self, init=False, intermediate=False, FLAG=False):
        training_output1 = Namespace()
        training_output1.av = self.av
        training_output1.trained_model = self.trained_model

        training_output2 = Namespace()
        training_output2.av = self.av
        training_output2.trained_perm_model = self.trained_perm_model

        config = configure(self.machine)

        if init == False and intermediate == False:
            path = config["save_path_for_trained_model"]
            os.makedirs(path, exist_ok=True)
            filename1 = (
                path
                + self.task
                + "_"
                + self.phi_concave_func
                + "_"
                + self.set_func_str
                + "_output"
            )
            filename2 = (
                path
                + self.task
                + "_"
                + self.phi_concave_func
                + "_"
                + self.set_func_str
                + "_output_perm"
            )

        if init == True and FLAG == False:
            training_output1.trained_model = None
            training_output1.init_model = self.init_model

            training_output2.trained_model = None
            training_output2.init_perm_model = self.init_perm_model

            path = config["save_path_for_initial_model"]
            os.makedirs(path, exist_ok=True)
            filename1 = (
                path
                + self.task
                + "_"
                + self.phi_concave_func
                + "_"
                + self.set_func_str
                + "_output_init_Fixed"
            )
            filename2 = (
                path
                + self.task
                + "_"
                + self.phi_concave_func
                + "_"
                + self.set_func_str
                + "_output_init_Fixed_perm"
            )

        if init == True and FLAG == True:
            training_output1.trained_model = None
            training_output1.init_model = self.init_model

            training_output2.trained_model = None
            training_output2.init_perm_model = self.init_perm_model

            path = config["save_path_for_initial_model"]
            os.makedirs(path, exist_ok=True)
            filename1 = (
                path
                + self.task
                + "_"
                + self.phi_concave_func
                + "_"
                + self.set_func_str
                + "_output_init_FLAGGED"
            )
            filename2 = (
                path
                + self.task
                + "_"
                + self.phi_concave_func
                + "_"
                + self.set_func_str
                + "_output_init_FLAGGED_perm"
            )

        if intermediate == True:
            path = config["save_path_for_intermediate_model"]
            os.makedirs(path, exist_ok=True)
            filename1 = (
                path
                + self.task
                + "_"
                + self.phi_concave_func
                + "_"
                + self.set_func_str
                + "_output_intermediate_epoch_"
                + str(self.current_epoch)
            )
            filename2 = (
                path
                + self.task
                + "_"
                + self.phi_concave_func
                + "_"
                + self.set_func_str
                + "_output_intermediate_epoch_perm"
                + str(self.current_epoch)
            )

        if self.av.Notebook == True:
            filename1 = filename1 + "_Notebook"
            filename2 = filename2 + "_Notebook"

        save_into_pickle(training_output1, filename1)
        save_into_pickle(training_output2, filename2)

    def read_init_model(self):
        config = configure(self.machine)
        path = config["save_path_for_initial_model"]
        filename = (
            path
            + self.task
            + "_"
            + self.phi_concave_func
            + "_"
            + self.set_func_str
            + "_output_init_Fixed"
        )
        if self.av.Notebook == True:
            filename = filename + "_Notebook"
        training_output = read_from_pickle(filename)
        return training_output.init_model

    def read_init_perm_model(self):
        config = configure(self.machine)
        path = config["save_path_for_initial_model"]
        filename = (
            path
            + self.task
            + "_"
            + self.phi_concave_func
            + "_"
            + self.set_func_str
            + "_output_init_Fixed_perm"
        )

        if self.av.Notebook == True:
            filename = filename + "_Notebook"

        training_output = read_from_pickle(filename)
        return training_output.init_perm_model

    def count_params(self):
        c = 0
        for p in self.init_model.parameters():
            c = c + torch.numel(p)

        return c


class evaluate_trained_model:
    def __init__(self, av):
        MakeCopy(av, self)
        config = configure(self.machine)
        path = config["save_path_for_trained_model"]

        filename1 = (
            path + self.task + "_" + self.phi_concave_func + "_" + self.set_func_str + "_output"
        )

        if av.Notebook == True:
            filename1 = filename1 + "_Notebook"

        training_output = read_from_pickle(filename1)
        self.model = training_output.trained_model
        self.task = av.task

    def predict(self, ValidationData, TestData):
        self.model = self.model.to(self.device)
        self.model.eval()

        self.U_test = TestData["U_test"]
        self.S_test = TestData["S_test"]

        self.U_dev = ValidationData["U_dev"]
        self.S_dev = ValidationData["S_dev"]

        self.metrics_test = evaluate_all_metrics(self.U_test, self.S_test, self.model)
        self.metrics_dev = evaluate_all_metrics(self.U_dev, self.S_dev, self.model)
        self.jc_test, self.map_test = self.metrics_test["mjc"], self.metrics_test["map"]
        self.jc_dev, self.map_dev = self.metrics_dev["mjc"], self.metrics_dev["map"]

    def compute_error(self):
        return self.jc_dev, self.map_dev, self.jc_test, self.map_test

    def extra_metrics(self):
        """NDCG@10, the paper's Jaccard@10 and NDCG, and standard errors over registries."""
        return {
            f"{k}_{split}": v
            for split, m in (("dev", self.metrics_dev), ("test", self.metrics_test))
            for k, v in m.items()
            if k not in ("mjc", "map")
        }
