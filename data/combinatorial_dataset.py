import argparse
from collections.abc import Sequence

from torch.utils.data import Dataset

from data.generate_combinatorial_data import synthetic_dataset


class CombinatorialDataset(Dataset):
    def __init__(self, dataset_dict: dict, dataset_type: int):
        self.data_tensor_padded_folds = dataset_dict["data_tensor_padded"][dataset_type]
        self.F_S_folds = dataset_dict["F_S_value"][dataset_type]
        self.n = self.data_tensor_padded_folds.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.data_tensor_padded_folds[index], self.F_S_folds[index]


def create_combinatorial_dataset(
    device: str,
    num_features: int,
    total_number_of_elements_in_ground_set: int,
    num_samples: int,
    max_set_size: int,
    phi_concave_func: str,
    folds: Sequence[float],
) -> tuple[CombinatorialDataset, CombinatorialDataset, CombinatorialDataset]:
    dataset_object = synthetic_dataset(
        argparse.Namespace(
            device=device,
            num_features=num_features,
            total_number_of_elements_in_ground_set=total_number_of_elements_in_ground_set,
            num_data_pts=num_samples,
            max_set_size=max_set_size,
            phi_concave_func=phi_concave_func,
            folds=folds,
        )
    )
    dataset_object.create_dataset()
    dataset_dict = dataset_object.split_into_training_test()
    train_dataset = CombinatorialDataset(dataset_dict, dataset_type=0)
    val_dataset = CombinatorialDataset(dataset_dict, dataset_type=1)
    test_dataset = CombinatorialDataset(dataset_dict, dataset_type=2)
    return train_dataset, val_dataset, test_dataset
