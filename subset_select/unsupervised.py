
import submodlib.functions as subf
import torch
from sklearn.metrics import ndcg_score, pairwise_distances

from common.utils import (
    jaccard_similarity,
)


def compute_precision_acc(labels, predictions):
    sorted_predictions = sorted(((e, i) for i, e in enumerate(predictions)), reverse=True)
    precision = 0.0
    r = int(sum(labels))
    j = 1
    for i in range(len(sorted_predictions)):
        if labels[sorted_predictions[i][1]] == 1.0:
            precision += 1.0 * j / (i + 1) / r
            j += 1

    return precision


def unionx(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list


class baseline:
    def __init__(self, av, U, S):
        methodDict = {
            "fac_loc": subf.facilityLocation.FacilityLocationFunction,
            "DPP": subf.logDeterminant.LogDeterminantFunction,
            "DM": subf.disparityMin.DisparityMinFunction,
        }
        self.av = av
        self.method = av.method
        self.num_of_U_S = len(U)
        self.U = U
        self.ArgMax_list = S
        self.data_mat = []
        self.func = []
        for i in range(self.num_of_U_S):
            data_mat = U[i]
            self.data_mat.append(data_mat)

            number_of_elements_ground_set = U[i].shape[0]
            DataMat = data_mat.numpy()
            S = 1 - pairwise_distances(DataMat, metric="cosine")
            if self.method == "fac_loc":
                func = methodDict[self.method](
                    number_of_elements_ground_set, "dense", separate_rep=False, sijs=S
                )
            if self.method == "DPP":
                func = methodDict[self.method](number_of_elements_ground_set, "dense", 1, sijs=S)

            if self.method == "DM":
                func = methodDict[self.method](number_of_elements_ground_set, "dense", sijs=S)
            self.func.append(func)

    def compute_ArgMax(self, i, budget):
        ArgMax_Indx = self.func[i].maximize(budget)
        return torch.tensor(ArgMax_Indx)[:, 0].long().tolist()

    def evaluate_standalone_from_argmax(self):
        JC = []
        AP = []
        ArgMax_list = self.ArgMax_list
        for i in range(len(ArgMax_list)):
            ArgMax_est = self.compute_ArgMax(i, 10)
            jc = jaccard_similarity(ArgMax_est, ArgMax_list[i].tolist())
            candidates = unionx(ArgMax_est, ArgMax_list[i].tolist())
            labels = []
            predictions = []
            for j in range(len(candidates)):
                if candidates[j] in ArgMax_list[i].tolist():
                    labels.append(1.0)
                else:
                    labels.append(0.0)

                if candidates[j] in ArgMax_est:
                    position = ArgMax_est.index(candidates[j])
                    predictions.append(1 / (position + 1))
                else:
                    predictions.append(0.0)
            JC.append(jc)

            AP.append(ndcg_score([labels], [predictions]))
        return JC, AP, sum(JC) / len(JC), sum(AP) / len(AP)
