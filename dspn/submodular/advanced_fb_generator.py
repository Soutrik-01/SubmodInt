import sys

sys.path.append("..")
from tqdm import tqdm
import numpy as np

from .optimizer import DSF_opt, DSF_opt_set_transformer
from .fl import get_features
import pickle as pkl
import os 

def generate_balanced_random_summary(V ,B, Y, K=100):
    V = np.arange(V)
    budgets = np.ones(K)*(B//K) 
    if B%K != 0 :
        idx = np.random.choice(K, B%K, replace=False)
        budgets[idx] += 1
    assert budgets.sum() == B

    summary = []
    for jj in range(K):
        summary.append(np.random.choice(V[Y == jj], int(budgets[jj]), replace=False))
    summary = np.concatenate(summary)
    return summary



def dewrap(model, ddp=False):
    if ddp:
        return model.module
    else:
        return model

def balanced_random_selection(B, Y, K):
    V = np.arange(K)
    budgets = np.ones(K)*(B//K) 
    if B%K != 0 :
        idx = np.random.choice(K, B%K, replace=False)
        budgets[idx] += 1
    sset = []
    for i in range(K):
        if budgets[i] == 0:
            continue 
        sset.append(np.random.choice(V[Y==i], int(budgets[i]), replace=False))
    sset = np.concatenate(sset)
    
    return sset         

class DiverseDataGenerator:
    """docstring for DataGenerator"""

    def __init__(
        self,
        X,
        Y,
        model,
        args,
        dataset_,
        device,
        budget=1,
        set_size=20,
        K=100,
    ):

        self.budget = budget
        self.set_size = set_size
        self.device = device
        self.args = args
        self.K = K
        self.X = X
        self.Y = np.asarray(Y)
        self.model = model
        self.V_partition_to_gs = {}
        if args['model_type'] == 'set_transformer':    
            H = None 
        else:
            H = get_features(
                dewrap(model, True).feat,
                X,
                dewrap(model, True).device,
            )
            H = H.squeeze()
        self.diverse_sets, self.intermediate_sets = self.get_diverse_sets(H, X)
        if args["nnkmeans_feedback"]:
            self.args["nnkmeans_path"] = os.path.join(os.path.join(args["root"], args["dset"]), "feedback_idx/nnkmeans_feedback_summary.npy")
            self.nnkmeans_set = np.load(self.args["nnkmeans_path"])
        else:
            self.args["nnkmeans_path"]=None
        if args["target_feedback"]:
            self.args["flsummary_path"] = os.path.join(os.path.join(args["root"], args["dset"]), "feedback_idx/fl_feedback_summary.npy")
            self.flsummary_set = np.load(self.args["flsummary_path"])
        else:
            self.args["flsummary_path"]=None

    def get_diverse_sets(self, H, X):
        diverse_sets = {"coarse": [], 'balanced':[]}
        upper_lim = 95
        lower_lim = int(100 * self.args["p_min"])
        probs = np.asarray(range(upper_lim, lower_lim - 5, -5)) / 100
        self.probs = probs
        intermediate_sets = {i: [] for i in probs}

        print("Over full ground set...")
        for _ in tqdm(range(self.budget)):
            diverse_sets["coarse"].append(
                DSF_opt(dewrap(self.model, True), H).stochastic_greedy(
                    self.set_size,
                    self.device,
                    self.args["out_dim"],
                    r_size=self.args["r_size"],
                ) if self.args['model_type'] != 'set_transformer' else  DSF_opt_set_transformer(dewrap(self.model, True), X).stochastic_greedy(
                    self.set_size,
                    self.device,
                    self.args["out_dim"],
                    r_size=self.args["r_size"],
                )
            )
            if self.args["class_balanced_feedback"] or self.args["matroid_v_non_matroid"]:
                if self.args['model_type'] == 'set_transformer':
                    f = DSF_opt_set_transformer(dewrap(self.model, True), X).balanced_stochastic_greedy_max(
                            self.set_size,
                            self.device,
                            self.args["out_dim"],
                            self.Y,
                            self.K,
                            r_size=self.args["r_size"],
                        )
                else:    
                    f = DSF_opt(dewrap(self.model, True), H).balanced_stochastic_greedy_max(
                            self.set_size,
                            self.device,
                            self.args["out_dim"],
                            self.Y,
                            self.K,
                            r_size=self.args["r_size"],
                        )
                diverse_sets["balanced"].append(f) 

            for p in probs:
                intermediate_sets[p].append(
                    DSF_opt(
                        dewrap(self.model, True), H
                    ).stochastic_greedy_prob(
                        self.set_size,
                        self.device,
                        self.args["out_dim"],
                        probs=p,
                        r_size=self.args["r_size"],
                    ) if self.args['model_type'] != 'set_transformer' else DSF_opt_set_transformer(
                        dewrap(self.model, True), X
                    ).stochastic_greedy_prob(
                        self.set_size,
                        self.device,
                        self.args["out_dim"],
                        probs=p,
                        r_size=self.args["r_size"],
                    )
                )



        diverse_sets["coarse"] = np.asarray(diverse_sets["coarse"])
        if self.args["class_balanced_feedback"] or self.args["matroid_v_non_matroid"]:
            diverse_sets["balanced"] = np.asarray(diverse_sets["balanced"])
        print("After CB")
        for p in probs:
            intermediate_sets[p] = np.asarray(intermediate_sets[p])


        with open("diverse_sets.pkl", 'wb') as f:
            pkl.dump(diverse_sets, f)

        return diverse_sets, intermediate_sets

    def fetch_diverse_sets(
        self,
        dataset_,
        V,
        num_rand=None,
        D_M_idx_full_train=None,
        D_E_idx_full_train=None,
    ):

        D_M_idx_full = []
        D_E_idx_full = []


        if self.args["matroid_v_non_matroid"]:
            D_M_idx_full_matroid_v_nmatroid = []
            D_E_idx_full_matroid_v_nmatroid = []
        else:
            D_M_idx_full_matroid_v_nmatroid, D_E_idx_full_matroid_v_nmatroid = None, None

        if self.args["all_remaining"]:
            D_M_idx_full_all_remaining = []
            D_E_idx_full_all_remaining = []
        else:
            D_M_idx_full_all_remaining, D_E_idx_full_all_remaining = None, None


        if self.args["class_balanced_feedback"]:
            D_M_idx_full_balanced = []
            D_E_idx_full_balanced = []
        else:
            D_M_idx_full_balanced, D_E_idx_full_balanced = None, None

        if self.args["nnkmeans_path"]:
            D_M_idx_full_nnkmeans = []
            D_E_idx_full_nnkmeans = []
        else:
            D_M_idx_full_nnkmeans, D_E_idx_full_nnkmeans = None, None

        if self.args["flsummary_path"]:
            D_M_idx_full_fl = []
            D_E_idx_full_fl = []
        else:
            D_M_idx_full_fl, D_E_idx_full_fl = None, None



        num_rand_comparisions = num_rand if num_rand else len(D_M_idx_full)
        num_rand_comparisions = num_rand_comparisions // (1 + len(self.probs))
        diversity_idxes = np.random.choice(self.budget, num_rand_comparisions)

        if not (D_M_idx_full_train is None):
            em_indexes = np.random.choice(
                len(D_M_idx_full_train), num_rand_comparisions
            )

        for i in range(num_rand_comparisions):
            A = np.random.choice(V, self.args["set_size"], replace=False)

            D_M_idx_full.append(A)
            D_E_idx_full.append(self.diverse_sets["coarse"][diversity_idxes[i]])

            if self.args["class_balanced_feedback"]:
                A_cb = generate_balanced_random_summary(V, self.args["set_size"], self.Y, self.K)
                D_M_idx_full_balanced.append(A_cb)
                D_E_idx_full_balanced.append(
                    self.diverse_sets["coarse"][diversity_idxes[i]]
                )

                assert D_M_idx_full_balanced[-1].shape == D_E_idx_full_balanced[-1].shape

                D_M_idx_full_balanced.append(A_cb)
                D_E_idx_full_balanced.append(
                    self.diverse_sets["balanced"][diversity_idxes[i]]
                )
                assert D_M_idx_full_balanced[-1].shape == D_E_idx_full_balanced[-1].shape


            if self.args["matroid_v_non_matroid"]:
                D_M_idx_full_matroid_v_nmatroid.append(
                    self.diverse_sets["coarse"][diversity_idxes[i]]
                )
                D_E_idx_full_matroid_v_nmatroid.append(
                    self.diverse_sets["balanced"][diversity_idxes[i]]
                )


            if self.args["nnkmeans_path"]:
                D_M_idx_full_nnkmeans.append(
                    self.nnkmeans_set[i % len(self.nnkmeans_set)]
                )
                D_E_idx_full_nnkmeans.append(
                    self.diverse_sets["coarse"][diversity_idxes[i]]
                )

            if self.args["flsummary_path"]:
                D_M_idx_full_fl.append(self.flsummary_set[i % len(self.flsummary_set)])
                D_E_idx_full_fl.append(self.diverse_sets["coarse"][diversity_idxes[i]])

            if self.args["all_remaining"]:
                D_E_idx_full_all_remaining.append(self.flsummary_set[i % len(self.flsummary_set)])
                D_M_idx_full_all_remaining.append(A)

                D_E_idx_full_all_remaining.append(self.flsummary_set[i % len(self.flsummary_set)])
                D_M_idx_full_all_remaining.append(A_cb)

                D_M_idx_full_all_remaining.append(self.flsummary_set[i % len(self.flsummary_set)])
                D_E_idx_full_all_remaining.append(self.diverse_sets["balanced"][diversity_idxes[i]])

                D_E_idx_full_all_remaining.append(self.nnkmeans_set[i % len(self.nnkmeans_set)])
                D_M_idx_full_all_remaining.append(A)

                D_E_idx_full_all_remaining.append(self.nnkmeans_set[i % len(self.nnkmeans_set)])
                D_M_idx_full_all_remaining.append(A_cb)

                D_M_idx_full_all_remaining.append(self.nnkmeans_set[i % len(self.nnkmeans_set)])
                D_E_idx_full_all_remaining.append(self.diverse_sets["balanced"][diversity_idxes[i]])


            if not (D_M_idx_full_train is None):
                D_M_idx_full.append(D_M_idx_full_train[em_indexes[i]])
                D_E_idx_full.append(self.diverse_sets["coarse"][diversity_idxes[i]])
                assert D_M_idx_full[-1].shape == D_E_idx_full[-1].shape
                D_M_idx_full.append(D_E_idx_full_train[em_indexes[i]])
                D_E_idx_full.append(self.diverse_sets["coarse"][diversity_idxes[i]])
                assert D_M_idx_full[-1].shape == D_E_idx_full[-1].shape

            for k in self.intermediate_sets.keys():
                D_M_idx_full.append(A)
                D_E_idx_full.append(self.intermediate_sets[k][diversity_idxes[i]])
                if not (D_M_idx_full_train is None):
                    D_M_idx_full.append(D_M_idx_full_train[em_indexes[i]])
                    D_E_idx_full.append(self.intermediate_sets[k][diversity_idxes[i]])
                    assert D_M_idx_full[-1].shape == D_E_idx_full[-1].shape
                    D_M_idx_full.append(D_E_idx_full_train[em_indexes[i]])
                    D_E_idx_full.append(self.intermediate_sets[k][diversity_idxes[i]])
                    assert D_M_idx_full[-1].shape == D_E_idx_full[-1].shape

                if self.args["nnkmeans_path"]:
                    D_M_idx_full_nnkmeans.append(
                        self.nnkmeans_set[i % len(self.nnkmeans_set)]
                    )
                    D_E_idx_full_nnkmeans.append(
                        self.intermediate_sets[k][diversity_idxes[i]]
                    )

                if self.args["flsummary_path"]:
                    D_M_idx_full_fl.append(
                        self.flsummary_set[i % len(self.flsummary_set)]
                    )
                    D_E_idx_full_fl.append(
                        self.intermediate_sets[k][diversity_idxes[i]]
                    )

        print(len(D_M_idx_full), len(D_E_idx_full))
        D_M_idx_full, D_E_idx_full = np.stack(D_M_idx_full), np.stack(D_E_idx_full)
        idx = np.random.permutation(len(D_M_idx_full))
        D_M_idx_full, D_E_idx_full = D_M_idx_full[idx], D_E_idx_full[idx]


        if self.args["nnkmeans_path"]:
            D_M_idx_full_nnkmeans, D_E_idx_full_nnkmeans = np.stack(
                D_M_idx_full_nnkmeans
            ), np.stack(D_E_idx_full_nnkmeans)
            idx = np.random.permutation(len(D_M_idx_full_nnkmeans))
            D_M_idx_full_nnkmeans, D_E_idx_full_nnkmeans = (
                D_M_idx_full_nnkmeans[idx],
                D_E_idx_full_nnkmeans[idx],
            )

        if self.args["flsummary_path"]:
            D_M_idx_full_fl, D_E_idx_full_fl = np.stack(D_M_idx_full_fl), np.stack(
                D_E_idx_full_fl
            )
            idx = np.random.permutation(len(D_M_idx_full_fl))
            D_M_idx_full_fl, D_E_idx_full_fl = (
                D_M_idx_full_fl[idx],
                D_E_idx_full_fl[idx],
            )


        if self.args["class_balanced_feedback"]:
            D_M_idx_full_balanced, D_E_idx_full_balanced = np.stack(D_M_idx_full_balanced), np.stack(
                D_E_idx_full_balanced
            )
            idx = np.random.permutation(len(D_M_idx_full_balanced))
            D_M_idx_full_balanced, D_E_idx_full_balanced = (
                D_M_idx_full_balanced[idx],
                D_E_idx_full_balanced[idx],
            )

        if self.args["matroid_v_non_matroid"]:
            D_M_idx_full_matroid_v_nmatroid, D_E_idx_full_matroid_v_nmatroid = np.stack(D_M_idx_full_matroid_v_nmatroid), np.stack(
                D_E_idx_full_matroid_v_nmatroid
            )
            idx = np.random.permutation(len(D_M_idx_full_matroid_v_nmatroid))
            D_M_idx_full_matroid_v_nmatroid, D_E_idx_full_matroid_v_nmatroid = (
                D_M_idx_full_matroid_v_nmatroid[idx],
                D_E_idx_full_matroid_v_nmatroid[idx],
            )

        if self.args["all_remaining"]:
            D_M_idx_full_all_remaining, D_E_idx_full_all_remaining = np.stack(D_M_idx_full_all_remaining), np.stack(
                D_E_idx_full_all_remaining
            )
            idx = np.random.permutation(len(D_M_idx_full_all_remaining))
            D_M_idx_full_all_remaining, D_E_idx_full_all_remaining = (
                D_M_idx_full_all_remaining[idx],
                D_E_idx_full_all_remaining[idx],
            )


        np.save("D_M_idx_full.npy", D_M_idx_full)
        np.save("D_E_idx_full.npy", D_E_idx_full)


        np.save("D_M_idx_full_nnkmeans.npy", D_M_idx_full_nnkmeans)
        np.save("D_E_idx_full_nnkmeans.npy", D_E_idx_full_nnkmeans)

        np.save("D_M_idx_full_fl.npy", D_M_idx_full_fl)
        np.save("D_E_idx_full_fl.npy", D_E_idx_full_fl)

        np.save("D_M_idx_full_matroid_v_nmatroid.npy", D_M_idx_full_matroid_v_nmatroid)
        np.save("D_E_idx_full_matroid_v_nmatroid.npy", D_E_idx_full_matroid_v_nmatroid)

        np.save("D_M_idx_full_balanced.npy", D_M_idx_full_balanced)
        np.save("D_E_idx_full_balanced.npy", D_E_idx_full_balanced)

        np.save("D_M_idx_full_all_remaining.npy", D_M_idx_full_all_remaining)
        np.save("D_E_idx_full_all_remaining.npy", D_E_idx_full_all_remaining)

        return (
            D_M_idx_full,
            D_E_idx_full,
            D_M_idx_full_nnkmeans,
            D_E_idx_full_nnkmeans,
            D_M_idx_full_fl,
            D_E_idx_full_fl,
            D_M_idx_full_balanced,
            D_E_idx_full_balanced,
            D_M_idx_full_matroid_v_nmatroid,
            D_E_idx_full_matroid_v_nmatroid,
            D_M_idx_full_all_remaining,
            D_E_idx_full_all_remaining
        )