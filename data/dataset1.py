import torch
import torch.utils.data
import os
import logging
import numpy as np
import pandas as pd


class MTSFDataset(torch.utils.data.Dataset):
    """Multi-variate Time-Series Dataset for *.txt file

    Returns:
        [sample, label]
    """

    def __init__(
        self,
        window,
        horizon,
        data_name="wecar",
        set_type="train",  # 'train'/'validation'/'test'
        data_dir="./data",
    ):
        assert type(set_type) == type("str")
        self.window = window
        self.horizon = horizon
        self.data_dir = data_dir
        self.set_type = set_type

        # file_path = os.path.join(
        #     data_dir, data_name, "{}_{}.txt".format(data_name, set_type)
        # )
        dataPath = "E:/Github-Repositories/datasets/truedata/electricity/"
        if set_type == "train":
            rawdata = pd.read_csv(
                dataPath + "electricity.txt", sep=",", header=None
            ).values[:700]
            print('\n train')
        elif set_type == "validation":
            rawdata = pd.read_csv(
                dataPath + "electricity.txt", sep=",", header=None
            ).values[700:800]
            print('\n val')
            
        elif set_type == "test":
            rawdata = pd.read_csv(
                dataPath + "electricity.txt", sep=",", header=None
            ).values[800:1000]
            print('\n test')
            

        # rawdata = np.loadtxt(open(file_path), delimiter=",")
        self.len, self.var_num = rawdata.shape
        self.sample_num = max(self.len - self.window - self.horizon + 1 - 1, 0)
        self.samples, self.labels = self.__getsamples(rawdata)

    def __getsamples(self, data):
        X = torch.zeros((self.sample_num, self.window, self.var_num))
        Y = torch.zeros((self.sample_num, 1, self.var_num))

        for i in range(self.sample_num):
            start = i
            end = i + self.window
            X[i, :, :] = torch.from_numpy(data[start:end, :])
            Y[i, :, :] = torch.from_numpy(data[end + self.horizon - 1, :])

        return (X, Y)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        sample = [self.samples[idx, :, :], self.labels[idx, :, :]]

        return sample
