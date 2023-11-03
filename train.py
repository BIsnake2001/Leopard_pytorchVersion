import pyBigWig
import argparse
import os
import sys
import numpy as np
import re
from dataset import UNetDataModule
from unet import LitUNetFT
from unet_model import UNet
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint



def get_args():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument('-tf', '--transcription_factor', default='CTCF', type=str,
        help='transcript factor')
    parser.add_argument('-tr', '--train', default='K562', type=str,
        help='train cell type')
    parser.add_argument('-vali', '--validate', default='A549', type=str,
        help='validate cell type')
    parser.add_argument('-par', '--partition', default='1', type=str,
        help='chromasome parition')
    parser.add_argument('-batchsize', '--batchsize', default=32, type=int,
        help='batch size')
    args = parser.parse_args()
    return args

args=get_args()

print(sys.argv)
the_tf=args.transcription_factor
cell_train=args.train
cell_vali=args.validate
par=args.partition
batch_size = args.batchsize

## random seed for chr partition

#############################################

path_computer='../../data/'
path2=path_computer + 'dnase_bigwig/' # dnase
path3=path_computer + 'chipseq_conservative_refine_bigwig/' # label


# open bigwig
feature_avg=pyBigWig.open(path2 + 'avg.bigwig')
feature_train=pyBigWig.open(path2 + cell_train + '.bigwig')
feature_vali=pyBigWig.open(path2 + cell_vali + '.bigwig')
label_train=pyBigWig.open(path3 + the_tf + '_' + cell_train + '.bigwig')
label_vali=pyBigWig.open(path3 + the_tf + '_' + cell_vali + '.bigwig')
############

data_module = UNetDataModule(batch_size, par, feature_train, label_train, feature_vali, label_vali)
data_module.setup()

unet_model = UNet()
model = LitUNetFT(unet_model)


# configure trainer
callback_checkoint = ModelCheckpoint(save_top_k = 3, monitor = "valid_loss", mode = "min", filename = "{epoch}-{step}-{valid_loss:.4f}", save_last = True,every_n_epochs=1,save_weights_only = True)
trainer = pl.Trainer(
    max_epochs=5, log_every_n_steps=1,
    limit_val_batches=10, val_check_interval=128,
    accumulate_grad_batches=128, accelerator="gpu",
    fast_dev_run=False, precision="bf16-mixed",strategy="auto",
    callbacks=[
        LearningRateMonitor(logging_interval="step"),
        callback_checkoint,
    ],
    logger=pl.loggers.TensorBoardLogger("lightning_logs", name=args.name),
    )
print('3. training begins')
trainer.fit(
    model, data_module,
)
