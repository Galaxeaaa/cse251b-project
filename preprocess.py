import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

class ArgoverseDataset(Dataset):
    """Dataset class for Argoverse"""
    def __init__(self, data_path, city_idx_path):
        super(ArgoverseDataset, self).__init__()
        self.data_path = data_path
        self.city_idx = city_idx_path
        
    def __len__(self):
        return len(self.pkl_list)

    def __getitem__(self, idx):

        pkl_path = self.pkl_list[idx]
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        if self.transform:
            data = self.transform(data)

        return data

MIA_list = pickle.load("./MIA.pkl")
PIT_list = pickle.load("./PIT.pkl")

