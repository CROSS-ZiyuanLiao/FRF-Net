import os
import torch

from torch.utils.data.dataset import Dataset


class FRFDataset(Dataset):
    """
    Data set of frequency response functions.

    Parameters
    ----------
    configs : Configs (see configs.py)

    split : str
        'train' or 'test', specifying the usage of the data set.

    """
    _SPLIT_LIST = ('train', 'test')

    def __init__(self, configs, split):
        self.configs = configs

        # check split validity
        if split not in self._SPLIT_LIST:
            raise ValueError('Unknown split: \'{}\''.format(split))

        # train dataset
        if split == 'train':
            data_dir = configs.data_dir
            n_sample = configs.n_train_samples

        # test dataset
        if split == 'test':
            data_dir = configs.test_data_dir
            n_sample = configs.n_test_samples

        # sample file directory
        data_dir = os.path.join(data_dir, split)
        self.data_dir = data_dir

        # sample file list
        sample_file_list = os.listdir(data_dir)
        if n_sample != 'all':
            sample_file_list = sample_file_list[0: n_sample]
        self.sample_file_list = sample_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, key):
        sample_file = self.sample_file_list[key]
        sample_file_path = os.path.join(self.data_dir, sample_file)
        target, data = self._read_sample(sample_file_path)
        return target, data

    def _read_sample(self, sample_file_path):
        configs = self.configs

        with open(sample_file_path) as f:
            line = f.readline()
            multi_label = configs.multi_label
            label_list = configs.label_list

            # read labels and data, transfer labels into target
            if multi_label > 1:
                labels = [item for item
                          in line.split(',')[0: multi_label]]
                target = [0. for _ in range(configs.out_classes)]
                for label in labels:
                    if label != 0:
                        target[label_list.index(label)] = 1.
                data = [float(item) for item in line.split(',')[multi_label:]]
            else:
                label = line.split(',')[0]
                target = label_list.index(label)
                data = [float(item) for item in line.split(',')[1:]]

            # transfer target and data into torch.Tensor
            target = torch.tensor(target)
            data = torch.tensor(data).reshape(-1)

        return target, data
