import torch
import torch.nn as nn
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import configparser
from common.utils import read_from_pickle, scale_tensor, all_pairs_dist, configure, Namespace, save_into_pickle, MakeTranspose3DTensor
from matplotlib import pyplot as plt
import submodlib.functions as subf
from sklearn.metrics import pairwise_distances
import numpy as np
from typing import Tuple, List

subm_dict={'torch_log':torch.log, 'torch_sqrt': torch.sqrt, 'fac_loc':'fac_loc',
           'DPP':'DPP','DisSum':'DisSum','Gcut':'Gcut', 'LOG_SQRT': 'LOG_SQRT', 'LOG_DPP': 'LOG_DPP', 'Gcut0.8': 'Gcut0.8'}

def load_synthetic_dataset(av):
    machine = av.machine
    config = configure(machine)

    generated_data_filename = config['synthetic_data_path'] + av.generated_data_filename + '_' + av.phi_concave_func

    dataset_ = read_from_pickle(generated_data_filename)

    return dataset_


def sample_indices_and_masks(
    N: int,
    m: int,
    k: int,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:

    assert k <= N, "k must be <= N"

    mask = torch.zeros((m, k), dtype=torch.long)
    indices_list = []

    for i in range(m):
        s = torch.randint(1, k + 1, (1,)).item()
        if s > 0:
            idx = torch.randperm(N)[:s]
            mask[i, :s] = torch.ones(s)
        else:
            idx = torch.empty(0, dtype=torch.long)
        indices_list.append(idx)

    return mask, indices_list


class synthetic_dataset(object):

    def __init__(self,av):

        self.av = av
        self.number_of_elements_ground_set =  av.total_number_of_elements_in_ground_set
        self.dim_element = av.num_features
        self.num_data_pts = av.num_data_pts
        self.max_set_size = av.max_set_size
        self.phi_concave_func =  subm_dict[av.phi_concave_func]
        self.folds = av.folds
        self.data_mat = torch.abs(torch.randn(self.number_of_elements_ground_set, self.dim_element))
        self.masks, self.indices = sample_indices_and_masks(self.number_of_elements_ground_set,
                                                    self.num_data_pts, self.max_set_size)

        if self.phi_concave_func  == 'Gcut':
            DataMat = self.data_mat.cpu().numpy()
            S = 1-pairwise_distances(DataMat, metric="cosine")
            self.Gcut =  subf.graphCut.GraphCutFunction(self.number_of_elements_ground_set,
                                                        'dense', 0.01, separate_rep=False, ggsijs=S)

    def create_dataset(self):

        number_of_elements_ground_set =  self.number_of_elements_ground_set
        dim_element =  self.dim_element
        num_data_pts = self.num_data_pts
        max_size = self.max_set_size

        data_mat_tensor = torch.zeros(max_size, dim_element, num_data_pts)
        for i in range(self.num_data_pts):
            set_data = self.data_mat[self.indices[i], :]
            set_size = self.indices[i].shape[0]
            assert set_data.shape[0] == set_size
            data_mat_tensor[:set_size, :, i] = set_data

        masks_transposed_expanded = self.masks.t().unsqueeze(1).expand(max_size, dim_element, num_data_pts)

        self.data_tensor_padded = torch.abs(torch.mul(masks_transposed_expanded ,data_mat_tensor))


        if self.phi_concave_func in ['Gcut']:

            F_S_list = [self.Gcut.evaluate(set(list(self.indices[i].cpu().numpy())))  for i in range(self.num_data_pts)]

            self.F_S_value = torch.tensor(F_S_list)
            self.F_S_value = scale_tensor(self.F_S_value, 10)

        if self.phi_concave_func == 'Gcut0.8':
            DataMat = self.data_mat.cpu().numpy()
            S = 1-pairwise_distances(DataMat, metric="cosine")
            Gcut =  subf.graphCut.GraphCutFunction(self.number_of_elements_ground_set,
                                                   'dense', 0.8, separate_rep=False, ggsijs=S)
            F_S_list = [Gcut.evaluate(set(list(self.indices[i].cpu().numpy())))  for i in range(self.num_data_pts)]

            self.F_S_value = torch.tensor(F_S_list)
            self.F_S_value = scale_tensor(self.F_S_value, 10)


        if self.phi_concave_func in [torch.log, torch.sqrt]:

            self.F_S_value = self.phi_concave_func(torch.sum(self.data_tensor_padded,dim=[0,1]))

            self.F_S_value = scale_tensor(self.F_S_value, 10)

        if self.phi_concave_func == 'LOG_SQRT':
            sum = torch.sum(self.data_tensor_padded, dim=[0, 1])
            self.F_S_value = torch.log(sum) * torch.sqrt(sum)
            self.F_S_value = scale_tensor(self.F_S_value, 10)

        if self.phi_concave_func == 'fac_loc':

            X = all_pairs_dist(self.data_mat, self.data_mat).to(self.av.device)

            self.F_S_value = torch.empty(self.num_data_pts)

            for i in range(self.num_data_pts):

                set_indices = self.indices[i]
                self.F_S_value[i] = torch.sum(torch.max(X[set_indices, :],0,keepdim=True)[0])

            self.F_S_value = scale_tensor(self.F_S_value, 10)

        if self.phi_concave_func == 'DPP':
            self.data_mat = self.data_mat/torch.norm(self.data_mat)
            X = torch.mm(self.data_mat, self.data_mat.t()).to(self.av.device)
            self.F_S_value = torch.empty(self.num_data_pts)

            for i in range(self.num_data_pts):

                set_indices = self.indices[i]
                DPP_mat = X[set_indices][:, set_indices]+ (1+1e-6)*torch.eye(len(set_indices)).to(self.av.device)
                self.F_S_value[i] = torch.log(torch.linalg.det(DPP_mat))
            self.F_S_value = scale_tensor(self.F_S_value, 10)

        if self.phi_concave_func == 'LOG_DPP':
            self.data_mat = self.data_mat/torch.norm(self.data_mat)
            X = torch.mm(self.data_mat, self.data_mat.t()).to(self.av.device)
            self.F_S_value = torch.empty(self.num_data_pts)

            for i in range(self.num_data_pts):

                set_indices = self.indices[i]
                DPP_mat = X[set_indices][:, set_indices]+ (1+1e-6)*torch.eye(len(set_indices)).to(self.av.device)
                self.F_S_value[i] = torch.log(torch.linalg.det(DPP_mat))
            self.F_S_value = torch.log(torch.sum(self.data_tensor_padded, dim=[0, 1])) * self.F_S_value
            self.F_S_value = scale_tensor(self.F_S_value, 10)

    def split_into_training_test(self):


        data_tensor_padded, F_S_value, folds =  self.data_tensor_padded, self.F_S_value, self.folds

        num_elem =  data_tensor_padded.shape[-1]
        tr_size = int(folds[0]* num_elem)
        dev_size = int((folds[1]+folds[0])* num_elem)
        test_size =   num_elem
        folds = self.folds

        permuted_index = torch.randperm(num_elem)
        self.fold_indices = {'train':permuted_index[0:tr_size],
                              'dev': permuted_index[tr_size:dev_size],
                              'test': permuted_index[dev_size:test_size],
                              'all':permuted_index}


        data_tensor_padded_training = data_tensor_padded[:,:,permuted_index[0:tr_size]]
        F_S_training = F_S_value[permuted_index[0:tr_size]]

        data_tensor_padded_dev = data_tensor_padded[:,:,permuted_index[tr_size:dev_size]]
        F_S_dev = F_S_value[permuted_index[tr_size:dev_size]]

        data_tensor_padded_test = data_tensor_padded[:,:,permuted_index[dev_size:test_size]]
        F_S_test = F_S_value[permuted_index[dev_size:test_size]]

        self.data_tensor_padded_folds = (data_tensor_padded_training,data_tensor_padded_dev,data_tensor_padded_test)
        self.F_S_folds = (F_S_training, F_S_dev, F_S_test)

        self.data_tensor_padded_folds = (MakeTranspose3DTensor(data_tensor_padded_training),
                                         MakeTranspose3DTensor(data_tensor_padded_dev),
                                         MakeTranspose3DTensor(data_tensor_padded_test))

        return {"data_tensor_padded": self.data_tensor_padded_folds, "F_S_value": self.F_S_folds}

    def save(self):

        syn_dataset = Namespace()
        syn_dataset.data_tensor_padded_folds = self.data_tensor_padded_folds
        syn_dataset.F_S_folds = self.F_S_folds
        machine = self.av.machine
        config = configure(machine)

        self.generated_data_filename = self.av.generated_data_filename + '_' + self.av.phi_concave_func
        save_file = config['synthetic_data_path'] + self.generated_data_filename

        save_into_pickle(syn_dataset, save_file)
