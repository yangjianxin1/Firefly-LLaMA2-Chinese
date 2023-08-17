from os.path import join
import os
from loguru import logger
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        return data

