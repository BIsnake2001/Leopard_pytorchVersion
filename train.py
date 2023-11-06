
import os
import sys
import numpy as np
import pyBigWig
import argparse

from unet import LitUNetFT
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import DatasetFromH5, DataModuleFromH5

def get_args():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument('-f', '--factor', default='CTCF', type=str,
        help='transcript factor')
    parser.add_argument('-t', '--train', default='K562', type=str,
        help='train cell type')
    parser.add_argument('-v', '--valid', default='A549', type=str,
        help='validate cell type')
    parser.add_argument('-d', '--data', default=".", type=str,
        help='where to find the data')
    parser.add_argument('-b', '--batchsize', default=128, type=int,
        help='batch size')
    parser.add_argument('-gpu', '--gpu', default="0", type=str,
        help='gpu to use')
    parser.add_argument("--fa",default="/shared/zhangyuxuan/data/annotation/hg38.fa",type=str,help="path to fasta file")
    parser.add_argument("--num_workers", default=4, type=int, help="number of workers")

    args = parser.parse_args()
    return args

def valid_args(args):
    
    train_path = os.path.join(args.data, args.factor, args.train, "train.h5")
    assert os.path.exists(train_path), f"train data not found at {train_path}"
    val_path = os.path.join(args.data, args.factor, args.valid, "val.h5")
    assert os.path.exists(val_path), f"val data not found at {val_path}"
    assert args.batchsize > 0, f"batchsize must be positive, got {args.batchsize}"

    args.train_path = train_path
    args.val_path = val_path
    args.name = f"{args.factor}_{args.train}_{args.valid}"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    return args

if __name__=="__main__": 
    args = valid_args(get_args())
    data_module = DataModuleFromH5(
        args.fa,
        args.train_path, 
        args.val_path, 
        args.val_path, # ignore test dataset
        batch_size = args.batchsize,
        max_num=100_000,
        num_workers=args.num_workers,
        )
    model = LitUNetFT()
    # configure trainer
    callback_checkoint = ModelCheckpoint(save_top_k = 3, monitor = "val_loss", mode = "min", filename = "{epoch}-{step}-{valid_loss:.4f}", save_last = True,every_n_epochs=1,save_weights_only = True)
    trainer = pl.Trainer(
        max_epochs=20, log_every_n_steps=1,
        limit_val_batches=64, val_check_interval=128,
        accumulate_grad_batches=1, accelerator="gpu",
        fast_dev_run=False, precision="bf16-mixed",strategy="auto",
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            callback_checkoint,
            EarlyStopping(monitor="val_loss", patience=3, mode="min", check_on_train_epoch_end=True)
        ],
        logger=pl.loggers.TensorBoardLogger("lightning_logs", name=args.name),
        )
    print('3. training begins')

    trainer.fit(
        model, data_module
    )

