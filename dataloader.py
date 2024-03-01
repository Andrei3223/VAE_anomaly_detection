import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# https://github.com/cbhua/swat-preprocess/blob/master/notebook/swat-2015-dataset.ipynb

class SwatDataset(Dataset):
    ''' Dataset class generator on SWaT dataset.
    Args:
        - path: <str> preprocessed dataset numpy file path
        - feature_idx: <list<int>> choose features you want to use by index
        - start_idx: <int> choose period you want to use by index
        - end_idx: <int> choose period you want to use by index
        - windows_size: <int> history length you want to use
        - sliding: <int> history window moving step
    '''

    def __init__(self, path,
                 feature_idx: list,
                 start_idx: int, 
                 end_idx: int, 
                 windows_size: int = 100,
                 sliding:int=1,
                 labels_path = None):
        self.data = np.load(path, allow_pickle=True).take(feature_idx, axis=1)[start_idx:end_idx]
        self.data = torch.Tensor(self.data)
        if labels_path is not None:
            labels = np.load(labels_path)[start_idx:end_idx]
        else:
            labels = None
        self.labels = labels
        self.windows_size = windows_size
        self.sliding = sliding

    def __len__(self):
        return int((self.data.shape[0] - self.windows_size) / self.sliding) - 1

    def __getitem__(self, index):
        '''
        Returns:
            input: <np.array> [num_feature, windows_size]
            output: <np.array> [num_feature]
        '''
        start = index * self.sliding
        end = index * self.sliding + self.windows_size

        if self.labels is None:
            return self.data[start:end, :], []  # self.data[end+1, :]
        else:
            return self.data[start:end, :], self.labels[start:end]  # self.data[end+1, :],