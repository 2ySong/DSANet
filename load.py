import lightning.pytorch as pl
import os
import numpy as np
import torch

from dsanet_model import DSANet
from dsanet_model import add_model_config
from data import MTSFDataset
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")

SEED = 7
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

root_dir = "/home/songzy/myDL/Projects3/DSANet"
log_dir = os.path.join(root_dir, "dsanet_logs")

config = add_model_config(root_dir)

model = DSANet(config)
csv_logger = pl.loggers.CSVLogger(save_dir="lightning_logs/csi", name="version_4")
trainer = pl.Trainer(
    logger=csv_logger,
    callbacks=[pl.callbacks.ModelCheckpoint(every_n_train_steps=60, save_top_k=-1)],
)

model = DSANet(config)
ckp1 = "lightning_logs/csi/version_3/checkpoints/epoch=19-step=4280.ckpt"
ckp2 = "lightning_logs/csi/version_4/checkpoints/epoch=39-step=8560.ckpt"
ckp3 = "lightning_logs/csi/version_0/checkpoints/epoch=4-step=1070.ckpt"
trainer.test(
    model=model,
    ckpt_path=ckp2,
)
