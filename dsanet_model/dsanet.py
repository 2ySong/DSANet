import os
import logging
import traceback
from collections import OrderedDict
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lightning.pytorch import LightningModule
from data import MTSFDataset
from dsanet_model.Layers import EncoderLayer, DecoderLayer
import argparse


class Single_Global_SelfAttn_Module(nn.Module):

    def __init__(
        self,
        window,
        n_multiv,
        n_kernels,
        w_kernel,
        d_k,
        d_v,
        d_model,
        d_inner,
        n_layers,
        n_head,
        drop_prob=0.1,
    ):
        """
        Args:

        window (int): the length of the input window size
        n_multiv (int): num of univariate time series
        n_kernels (int): the num of channels
        w_kernel (int): the default is 1
        d_k (int): d_model / n_head
        d_v (int): d_model / n_head
        d_model (int): outputs of dimension
        d_inner (int): the inner-layer dimension of Position-wise Feed-Forward Networks
        n_layers (int): num of layers in Encoder
        n_head (int): num of Multi-head
        drop_prob (float): the probability of dropout
        """

        super(Single_Global_SelfAttn_Module, self).__init__()

        self.window = window
        self.w_kernel = w_kernel
        self.n_multiv = n_multiv
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.conv2 = nn.Conv2d(1, n_kernels, (window, w_kernel))
        self.in_linear = nn.Linear(n_kernels, d_model)
        self.out_linear = nn.Linear(d_model, n_kernels)

        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=drop_prob)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, return_attns=False):

        x = x.view(-1, self.w_kernel, self.window, self.n_multiv)
        x2 = F.relu(self.conv2(x))
        x2 = nn.Dropout(p=self.drop_prob)(x2)
        x = torch.squeeze(x2, 2)
        x = torch.transpose(x, 1, 2)
        src_seq = self.in_linear(x)

        enc_slf_attn_list = []

        enc_output = src_seq

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        enc_output = self.out_linear(enc_output)
        return (enc_output,)


class Single_Local_SelfAttn_Module(nn.Module):

    def __init__(
        self,
        window,
        local,
        n_multiv,
        n_kernels,
        w_kernel,
        d_k,
        d_v,
        d_model,
        d_inner,
        n_layers,
        n_head,
        drop_prob=0.1,
    ):
        """
        Args:

        window (int): the length of the input window size
        n_multiv (int): num of univariate time series
        n_kernels (int): the num of channels
        w_kernel (int): the default is 1
        d_k (int): d_model / n_head
        d_v (int): d_model / n_head
        d_model (int): outputs of dimension
        d_inner (int): the inner-layer dimension of Position-wise Feed-Forward Networks
        n_layers (int): num of layers in Encoder
        n_head (int): num of Multi-head
        drop_prob (float): the probability of dropout
        """

        super(Single_Local_SelfAttn_Module, self).__init__()

        self.window = window
        self.w_kernel = w_kernel
        self.n_multiv = n_multiv
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.conv1 = nn.Conv2d(1, n_kernels, (local, w_kernel))
        self.pooling1 = nn.AdaptiveMaxPool2d((1, n_multiv))
        self.in_linear = nn.Linear(n_kernels, d_model)
        self.out_linear = nn.Linear(d_model, n_kernels)

        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=drop_prob)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, return_attns=False):

        x = x.view(-1, self.w_kernel, self.window, self.n_multiv)
        x1 = F.relu(self.conv1(x))
        x1 = self.pooling1(x1)
        x1 = nn.Dropout(p=self.drop_prob)(x1)
        x = torch.squeeze(x1, 2)
        x = torch.transpose(x, 1, 2)
        src_seq = self.in_linear(x)

        enc_slf_attn_list = []

        enc_output = src_seq

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        enc_output = self.out_linear(enc_output)
        return (enc_output,)


class AR(nn.Module):

    def __init__(self, window):

        super(AR, self).__init__()
        self.linear = nn.Linear(window, 1)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)
        return x


class DSANet(LightningModule):

    def __init__(self, config):
        """
        Pass in parsed HyperOptArgumentParser to the model
        """
        super(DSANet, self).__init__()
        self.config = config
        self.batch_size = config["batch_size"]
        self.window = config["window"]
        self.local = config["local"]
        self.n_multiv = config["n_multiv"]
        self.n_kernels = config["n_kernels"]
        self.w_kernel = config["w_kernel"]

        # hyperparameters of model
        self.d_model = config["d_model"]
        self.d_inner = config["d_inner"]
        self.n_layers = config["n_layers"]
        self.n_head = config["n_head"]
        self.d_k = config["d_k"]
        self.d_v = config["d_v"]
        self.drop_prob = config["drop_prob"]

        # build model
        self.__build_model()
        self.val_epoch_outs = []
        self.test_epoch_outs = []

    def __build_model(self):
        """
        Layout model
        """
        self.sgsf = Single_Global_SelfAttn_Module(
            window=self.window,
            n_multiv=self.n_multiv,
            n_kernels=self.n_kernels,
            w_kernel=self.w_kernel,
            d_k=self.d_k,
            d_v=self.d_v,
            d_model=self.d_model,
            d_inner=self.d_inner,
            n_layers=self.n_layers,
            n_head=self.n_head,
            drop_prob=self.drop_prob,
        )

        self.slsf = Single_Local_SelfAttn_Module(
            window=self.window,
            local=self.local,
            n_multiv=self.n_multiv,
            n_kernels=self.n_kernels,
            w_kernel=self.w_kernel,
            d_k=self.d_k,
            d_v=self.d_v,
            d_model=self.d_model,
            d_inner=self.d_inner,
            n_layers=self.n_layers,
            n_head=self.n_head,
            drop_prob=self.drop_prob,
        )

        self.ar = AR(window=self.window)
        self.W_output1 = nn.Linear(2 * self.n_kernels, 1)
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.active_func = nn.Tanh()

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        """
        sgsf_output, *_ = self.sgsf(x)
        slsf_output, *_ = self.slsf(x)
        sf_output = torch.cat((sgsf_output, slsf_output), 2)
        sf_output = self.dropout(sf_output)
        sf_output = self.W_output1(sf_output)

        sf_output = torch.transpose(sf_output, 1, 2)

        ar_output = self.ar(x)

        output = sf_output + ar_output

        return output

    def loss(self, labels, predictions):
        if self.config["criterion"] == "l1_loss":
            loss = F.l1_loss(predictions, labels)
        elif self.config["criterion"] == "mse_loss":
            loss = F.mse_loss(predictions, labels)
        return loss

    def training_step(self, data_batch, batch_idx):
        """
        Lightning calls this inside the training loop
        """
        # forward pass
        x, y = data_batch

        y_hat = self.forward(x)

        # calculate loss
        loss_val = self.loss(y, y_hat)

        output = OrderedDict({"loss": loss_val})
        self.log(
            "train_loss",
            loss_val,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return output

    def validation_step(self, data_batch, batch_i):
        """
        Lightning calls this inside the validation loop
        """
        x, y = data_batch

        y_hat = self.forward(x)

        loss_val = self.loss(y, y_hat)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        # if self.trainer.use_ddp:
        loss_val = loss_val.unsqueeze(0)
        # self.log(
        #     "val_loss",
        #     loss_val,
        #     on_step=False,
        #     prog_bar=True,
        #     logger=True,
        # )
        output = OrderedDict(
            {
                "val_loss": loss_val,
                "y": y,
                "y_hat": y_hat,
            }
        )
        self.val_epoch_outs.append(output)
        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def on_validation_epoch_end(self):
        """
        在验证结束时调用以聚合输出

        Args:
            outputs (list): 验证步骤中每个输出的列表

        Returns:
            dict: 包含以下键值对的字典
                - val_loss (float): 平均验证损失
                - RRSE (float): 根相对平方误差
                - CORR (float): 相关系数
                - MAE (float): 平均绝对误差
        """
        # 获取验证周期的输出列表
        outputs = self.val_epoch_outs
        # 初始化损失和
        loss_sum = 0
        # 遍历输出列表，累加验证损失
        for x in outputs:
            loss_sum += x["val_loss"].item()
        # 计算平均验证损失
        val_loss_mean = loss_sum / len(outputs)

        # 拼接所有验证步骤的预测值和真实值
        y = torch.cat(([x["y"] for x in outputs]), 0)
        y_hat = torch.cat(([x["y_hat"] for x in outputs]), 0)

        # 获取变量数
        num_var = y.size(-1)
        # 将y和y_hat的维度调整为(样本数, 变量数)
        y = y.view(-1, num_var)
        y_hat = y_hat.view(-1, num_var)
        # 获取样本数
        sample_num = y.size(0)

        # 计算预测值与真实值的差值
        y_diff = y_hat - y
        # 计算真实值的均值
        y_mean = torch.mean(y)
        # 计算真实值减去均值的差值
        y_translation = y - y_mean

        # 计算根相对平方误差
        val_rrse = torch.sqrt(torch.sum(torch.pow(y_diff, 2))) / torch.sqrt(
            torch.sum(torch.pow(y_translation, 2))
        )

        # 计算真实值和预测值的均值
        y_m = torch.mean(y, 0, True)
        y_hat_m = torch.mean(y_hat, 0, True)
        # 计算真实值和预测值减去各自均值的差值
        y_d = y - y_m
        y_hat_d = y_hat - y_hat_m
        # 计算相关系数的分子
        corr_top = torch.sum(y_d * y_hat_d, 0)
        # 计算相关系数的分母
        corr_bottom = torch.sqrt(
            (torch.sum(torch.pow(y_d, 2), 0) * torch.sum(torch.pow(y_hat_d, 2), 0))
        )
        # 计算相关系数
        corr_inter = corr_top / corr_bottom
        val_corr = (1.0 / num_var) * torch.sum(corr_inter)

        # 计算平均绝对误差
        val_mae = (1.0 / (sample_num * num_var)) * torch.sum(torch.abs(y_diff))

        # 计算NMSE
        mse = torch.mean(torch.pow(y_diff, 2), dim=0)
        # 计算实际值的方差
        variance_y = torch.mean(torch.pow(y_translation, 2), dim=0)
        # 计算NMSE
        nmse = mse / variance_y
        # 创建一个字典，用于存储验证指标
        tqdm_dic = {
            "val_loss": val_loss_mean,
            # "MSE": torch.mean(mse).item(),
            "RRSE": val_rrse.item(),
            "CORR": val_corr.item(),
            # "MAE": val_mae.item(),
            # "NMSE": torch.mean(nmse).item(),
        }

        self.log_dict(tqdm_dic, on_epoch=True, prog_bar=True)
        return tqdm_dic

    def test_step(self, batch, batch_idx):
        """
        测试步骤
        Args:
            batch (tuple): 包含输入和标签的元组
            batch_idx (int): 批次索引
        Returns:
            dict: 包含以下键值对的字典
                - y (Tensor): 真实值
                - y_hat (Tensor): 预测值
        """
        # 从批次中获取输入和标签
        x, y = batch
        # 调用模型，计算预测值
        y_hat = self.forward(x)
        # 计算损失
        loss = self.loss(y_hat, y)
        # 创建一个字典，用于存储输出
        output = {
            "y": y,
            "y_hat": y_hat,
            "test_loss": loss,
        }
        self.log("test_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        self.test_epoch_outs.append(output)
        return output

    def on_test_epoch_end(self) -> None:
        # 获取验证周期的输出列表
        outputs = self.test_epoch_outs
        # 初始化损失和
        loss_sum = 0
        # 遍历输出列表，累加验证损失
        for x in outputs:
            loss_sum += x["test_loss"].item()
        # 计算平均验证损失
        val_loss_mean = loss_sum / len(outputs)

        # 拼接所有验证步骤的预测值和真实值
        y = torch.cat(([x["y"] for x in outputs]), 0)
        y_hat = torch.cat(([x["y_hat"] for x in outputs]), 0)

        # 获取变量数
        num_var = y.size(-1)
        # 将y和y_hat的维度调整为(样本数, 变量数)
        y = y.view(-1, num_var)
        y_hat = y_hat.view(-1, num_var)
        # 获取样本数
        sample_num = y.size(0)

        # 计算预测值与真实值的差值
        y_diff = y_hat - y
        # 计算真实值的均值
        y_mean = torch.mean(y)
        # 计算真实值减去均值的差值
        y_translation = y - y_mean

        # 计算根相对平方误差
        val_rrse = torch.sqrt(torch.sum(torch.pow(y_diff, 2))) / torch.sqrt(
            torch.sum(torch.pow(y_translation, 2))
        )

        # 计算真实值和预测值的均值
        y_m = torch.mean(y, 0, True)
        y_hat_m = torch.mean(y_hat, 0, True)
        # 计算真实值和预测值减去各自均值的差值
        y_d = y - y_m
        y_hat_d = y_hat - y_hat_m
        # 计算相关系数的分子
        corr_top = torch.sum(y_d * y_hat_d, 0)
        # 计算相关系数的分母
        corr_bottom = torch.sqrt(
            (torch.sum(torch.pow(y_d, 2), 0) * torch.sum(torch.pow(y_hat_d, 2), 0))
        )
        # 计算相关系数
        corr_inter = corr_top / corr_bottom
        val_corr = (1.0 / num_var) * torch.sum(corr_inter)

        # 计算平均绝对误差
        val_mae = (1.0 / (sample_num * num_var)) * torch.sum(torch.abs(y_diff))

        # 计算NMSE
        mse = torch.mean(torch.pow(y_diff, 2), dim=0)
        # 计算实际值的方差
        variance_y = torch.mean(torch.pow(y_translation, 2), dim=0)
        # 计算NMSE
        nmse = mse / variance_y
        # 创建一个字典，用于存储验证指标
        tqdm_dic = {
            "test_loss": val_loss_mean,
            # "MSE": torch.mean(mse).item(),
            "RRSE": val_rrse.item(),
            "CORR": val_corr.item(),
            # "MAE": val_mae.item(),
            "NMSE": torch.mean(nmse).item(),
        }

        self.log_dict(tqdm_dic, on_epoch=True, prog_bar=True)
        return tqdm_dic

    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [
            scheduler
        ]  # It is encouraged to try more optimizers and schedulers here

    def __dataloader(self, train):
        # init data generators
        set_type = train
        dataset = MTSFDataset(
            window=self.config["window"],
            horizon=self.config["horizon"],
            data_name=self.config["data_name"],
            set_type=set_type,
            data_dir=self.config["data_dir"],
        )

        # when using multi-node we need to add the datasampler
        train_sampler = None
        batch_size = self.config["batch_size"]

        try:
            if self.on_gpu:
                train_sampler = DistributedSampler(dataset, rank=self.trainer.proc_rank)
                batch_size = batch_size // self.trainer.world_size  # scale batch size
        except Exception as e:
            pass

        should_shuffle = train_sampler is None
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
        )

        return loader

    def train_dataloader(self):
        # print("tng data loader called")
        return self.__dataloader(train="train")

    def val_dataloader(self):
        # print("val data loader called")
        return self.__dataloader(train="validation")

    def test_dataloader(self):
        # print("test data loader called")
        return self.__dataloader(train="test")
