import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np



class CSVDataset(Dataset):

    def __init__(self,file_path,transform=None,header=None):
        super(CSVDataset,self).__init__()
        self.data = pd.read_csv(file_path,header=header)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x = self.data.iloc[index].to_numpy().astype(np.float)
        if self.transform is not None:
            x = self.transform(x)
        return torch.FloatTensor(x)


