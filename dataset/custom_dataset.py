import os

import numpy as np
from torch.utils.data import Dataset

from config import common
from utils import io_utils


class CustomDataset(Dataset):
    def __init__(self, is_train=True):
        super(CustomDataset, self).__init__()

        if is_train:
            self.base_data_dir = common.train_data_dir
        else:
            self.base_data_dir = common.test_data_dir

        self.items = [os.path.splitext(file)[0] for file in os.listdir(self.base_data_dir)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        file_name = self.items[index]
        sample = io_utils.load_pickle(os.path.join(self.base_data_dir, file_name + '.pkl'))
        data, label = self.augment_data(sample)
        return data, label

    def augment_data(self, sample):
        data, label = sample
        return data, label


if __name__ == '__main__':
    dataset = CustomDataset()
    data, label = dataset.__getitem__(0)
    print(data, label)
