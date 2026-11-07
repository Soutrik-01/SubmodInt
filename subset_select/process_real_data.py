import csv

import torch

from common.utils import (
    Namespace,
    configure,
    read_from_pickle,
    save_into_pickle,
)


def load_real_dataset(av):
    machine = av.machine
    config = configure(machine)

    generated_data_filename = (
        config["real_data_path"] + av.generated_data_filename + "_" + av.phi_concave_func
    )

    dataset_ = read_from_pickle(generated_data_filename)

    return dataset_


def read_real_data(av):
    machine = av.machine
    config = configure(machine)

    data_filename = config["source_real"] + av.phi_concave_func
    dataset_ = read_from_pickle(data_filename)
    for i in range(len(dataset_)):
        dataset_[i + 1] = torch.tensor(dataset_[i + 1])
    random_item = dataset_[1]
    data_ = torch.zeros(len(dataset_) + 1, random_item.shape[0])

    for i in range(len(dataset_)):
        data_[i + 1, :] = dataset_[i + 1]

    prfx = "1_100_100_100_"
    sufx = "_regs.csv"
    filename = prfx + av.phi_concave_func + sufx
    csv_filename = config["source_real"] + filename
    with open(csv_filename, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_NONNUMERIC, quotechar="|")
        S = {}
        i = -1
        for row in reader:
            i = i + 1
            S[i] = torch.tensor([int(row[x]) for x in range(len(row))]).long()
    max_len = 1
    for i in range(len(S)):
        max_len = max(len(S[i]), max_len)
    return data_, S, max_len


class real_dataset:
    def __init__(self, av):
        self.av = av
        data, S, max_len = read_real_data(av)
        self.S_raw = S
        self.max_len = max_len
        self.data = data
        self.number_of_elements_ground_set = len(data)
        self.num_of_U_S = 1
        self.max_len = max_len
        self.dim_element = data[1].shape[0]
        self.folds = av.folds
        self.pad_S()

    def pad_S(self):
        S_list = []
        S = self.S_raw
        print(len(S))
        for i in range(len(S)):
            if self.av.pad == True and S[i].shape[0] > 2:
                a = (S[i], torch.zeros(self.max_len - len(S[i])).int())
                Svar = torch.cat(a, dim=0)
                S_list.append(Svar)

            if self.av.pad == False and S[i].shape[0] > 2:
                Svar = S[i]
                S_list.append(Svar)
        self.S = S_list
        print(len(S_list))
        self.data_mat = [self.data] * len(S_list)

    def split_into_training_test(self):
        data_mat, S, folds = self.data_mat, self.S, self.folds

        num_elem = len(data_mat)
        tr_size = int(folds[0] * num_elem)
        dev_size = int((folds[1] + folds[0]) * num_elem)
        test_size = num_elem

        U_train = data_mat[0:tr_size]
        U_dev = data_mat[tr_size:dev_size]
        U_test = data_mat[dev_size:test_size]

        S_train = S[0:tr_size]
        S_dev = S[tr_size:dev_size]
        S_test = S[dev_size:test_size]

        self.U_sets = (U_train, U_dev, U_test)
        self.S_sets = (S_train, S_dev, S_test)

    def save(self):
        syn_dataset = Namespace()
        syn_dataset.U_sets = self.U_sets
        syn_dataset.S_sets = self.S_sets
        syn_dataset.max_len = self.max_len
        machine = self.av.machine
        config = configure(machine)

        self.generated_data_filename = (
            self.av.generated_data_filename + "_" + self.av.phi_concave_func
        )
        save_file = config["real_data_path"] + self.generated_data_filename

        save_into_pickle(syn_dataset, save_file)
