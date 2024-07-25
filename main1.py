import os
import numpy as np
import torch


from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from dsanet_model import DSANet
from dsanet_model import add_model_config
from data import MTSFDataset
from torch.utils.data import DataLoader
from data import MTSFDataset2
import warnings

warnings.filterwarnings("ignore")

SEED = 7
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

root_dir = "/root/myDL/DSANet"
log_dir = os.path.join(root_dir, "dsanet_logs")

config = add_model_config(root_dir)

model = DSANet(config)
# print(model)
tb_logger = TensorBoardLogger(name="electricity", save_dir=config["log_dir"], version=0)

dataset = MTSFDataset2(
    data_dir=root_dir+'/data/',
    data_name='electricity.txt',
    batch_size=64,
    num_workers=8,
    context_length=48,
    prediction_length=12,
)

trainer = Trainer(
    devices=1,
    logger=tb_logger,
    max_epochs=1,
)

trainer.fit(model, dataset)
trainer.test(model, dataset)
