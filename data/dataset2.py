from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch
import lightning.pytorch as pl
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

class MTSFDataset2(pl.LightningDataModule):
    
    def __init__(self, 
                 data_dir: str = './', 
                 data_name: str = '',
                 batch_size: int = 64,
                 num_workers: int =8,
                 context_length: int = 48,
                 prediction_length: int = 12) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.data_name = data_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.context_length = context_length
        self.prediction_length = prediction_length
        

        
    def setup(self, stage = None):
        dataset = pd.read_csv(
            self.data_dir + self.data_name, sep= ',', header=0
        ).values
        self.var_num = dataset.shape[1]
        
        self.seq_len = len(dataset)
        
        if stage == 'fit' or stage is None:
            self.train_data = self.getSamples(dataset[:int(self.seq_len*0.7)])
            self.val_data = self.getSamples(dataset[int(self.seq_len*0.7):int(self.seq_len*0.9)])
        
        if stage == 'test':
            self.test_data = self.getSamples(dataset[int(self.seq_len*0.9):])
        
    def getSamples(self, data):
        sample_num = data.shape[0] - 100
        X = torch.zeros((sample_num, self.context_length, self.var_num))
        Y = torch.zeros((sample_num, self.prediction_length, self.var_num))

        for i in range(sample_num):
            start = i
            end = i + self.context_length
            X[i] = torch.from_numpy(data[start : end])
            Y[i] = torch.from_numpy(data[end : end+self.prediction_length])

        return (X,Y)
        
    def collate_fn(data, batch_idx):
        print(data[0])
        return {
            'inputs': torch.stack(x[0] for x in data[batch_idx]),
            'labels': torch.tensor(x[1] for x in data[batch_idx])
        }
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle= 1,collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle= 0,collate_fn=self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle= 0,collate_fn=self.collate_fn)
     

if __name__ == '__main__':
    
    data_dir = '',
    data_name = 'electricity.txt'
    
    dataset = MTSFDataset2(
        data_dir=data_dir,
        data_name=data_name,
        batch_size=64,
        num_workers=8,
        context_length=48,
        prediction_length=12
    )
    