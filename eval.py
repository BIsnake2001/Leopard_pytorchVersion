import os, sys
import time
import pathlib
import argparse

from tqdm import tqdm

import numpy as np
import pandas as pd

import torch 
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
import torchmetrics as tm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from unet import LitUNetFT
from dataset import DatasetFromH5, DataModuleFromH5


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--factor", required = True, type=str, help="transcript factor")
    parser.add_argument("-c", "--cellline", required = True, type=str, help="test cell line")
    parser.add_argument("-t", "--test", required = False, default = "test.h5", type=str, help="test data file name")
    parser.add_argument('-k', '--ckpt_dir', default="./lightning_logs", type=str, help='where to find the ckpt (dir created by lightning)')
    parser.add_argument('-v', '--version', default="0", type=str, help='version of the ckpt to use')

    parser.add_argument('-d', '--data', default=".", type=str,
        help='where to find the data')
    parser.add_argument('-b', '--batchsize', default=128, type=int,
        help='batch size')
    parser.add_argument('-gpu', '--gpu', default="0", type=str,
        help='gpu to use')
    parser.add_argument("-s","--size",default = None, type=int,help="window size")
    parser.add_argument("-r","--resolution",default="1",type=str,help="resolution of the data")
    parser.add_argument("--fa",default="/shared/zhangyuxuan/data/annotation/hg38.fa",type=str,help="path to fasta file")
    parser.add_argument("--num_workers", default=4, type=int, help="number of workers")
    return parser.parse_args()


def valid_args(args):
    
    test_path = os.path.join(args.data, args.factor, args.cellline, args.test)
    assert os.path.exists(test_path), f"train data not found at {test_path}"

    assert args.batchsize > 0, f"batchsize must be positive, got {args.batchsize}"

    args.test_path = test_path
    args.name = f"{args.factor}_{args.cellline}_{args.test}_version_{args.version}"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    assert args.resolution in ["1","200"]

    assert os.path.exists(args.ckpt_dir), f"ckpt dir not found at {args.ckpt_dir}"
    args.ckpt_dir = pathlib.Path(args.ckpt_dir)
    return args


def load_model(ckpt):
    model = LitUNetFT.load_from_checkpoint(ckpt,resolution = args.resolution, size = args.size)
    model.eval()
    model.cuda()
    model = model.bfloat16()
    return model

def predict(model,dl):
    predicts = []
    labels = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dl, total=len(dl))):
            x, y = batch["inputs"], batch["label"]
            x = x.cuda().to(torch.bfloat16)
            y = y.cuda().to(torch.int)
            y_hat = model(x)
            probs = F.sigmoid(y_hat).view(-1).cpu().detach()

            predicts.append(probs)
            labels.append(y.view(-1).cpu().detach())
    predicts = torch.cat(predicts)
    labels = torch.cat(labels)
    return predicts, labels

def eval(predicts, labels):
    predicts = predicts.cuda()
    labels = labels.cuda()
    metrics_auroc = tm.AUROC(task="binary", ignore_index=-1).cuda()
    metrics_auprc = tm.AveragePrecision(task="binary", ignore_index=-1).cuda()
    metrics_mcc = tm.MatthewsCorrCoef(task="binary", ignore_index=-1).cuda()
    metrics_f1 = tm.F1Score(task="binary", ignore_index=-1).cuda()
    metrics_precision = tm.Precision(task="binary", ignore_index=-1).cuda()
    metrics_recall = tm.Recall(task="binary", ignore_index=-1).cuda()


    score_auroc = metrics_auroc(predicts, labels).cpu().item()
    score_auprc = metrics_auprc(predicts, labels).cpu().item()
    score_mcc = metrics_mcc(predicts, labels).cpu().item()
    score_f1 = metrics_f1(predicts, labels).cpu().item()
    score_precision = metrics_precision(predicts, labels).cpu().item()
    score_recall = metrics_recall(predicts, labels).cpu().item()

    metrics_auroc.reset()
    metrics_auprc.reset()
    metrics_mcc.reset()
    metrics_f1.reset()
    metrics_precision.reset()
    metrics_recall.reset()

    predicts = predicts.cpu()
    labels = labels.cpu()
    # clear gpu memory
    torch.cuda.empty_cache()

    scores = {
        "auroc": score_auroc, 
        "auprc": score_auprc, 
        "mcc": score_mcc, 
        "f1": score_f1,
        "precision": score_precision,
        "recall": score_recall
        }
    return scores

def process(args):

    test_dataset = DatasetFromH5(args.test_path, fa = args.fa, max_num=100000)
    dl = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)

    list_scores = []
    list_predicts = []
    list_labels = []
    ckpts = []
    for ckpt in args.ckpt_dir.glob(f"{args.factor}*/version_{args.version}/checkpoints/last.ckpt"):
        sample = ckpt.parent.parent.parent.name
        model = load_model(ckpt)
        predicts, labels = predict(model, dl)
        scores = eval(predicts, labels)
        scores["sample"] = sample
        print(ckpt)
        print(scores)
        list_scores.append(scores)
        list_predicts.append(predicts)
        list_labels.append(labels)
        ckpts.append(ckpt)

    final_predicts = torch.stack(list_predicts).mean(dim=0)
    final_labels = list_labels[0]
    final_scores = eval(final_predicts, final_labels)
    final_scores["sample"] = "final"
    list_scores.append(final_scores)

    df_scores = pd.DataFrame(list_scores)
    return df_scores, ckpts

if __name__=="__main__":
    args = valid_args(get_args())
    df_scores, ckpts = process(args)
    with open(f"{args.name}.txt", "w") as f:
        f.write(f"#"*20+"\n")
        # current time
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        f.write(f"# {time_str}\n") 
        f.write(f"# name: {args.name}\n")
        model_params = f"resolution: {args.resolution}, size: {args.size}"
        f.write(f"# model: {model_params}\n")
        f.write(f"# test file: {args.test_path}\n")
        f.write(f"# ckpts: {args.ckpt_dir} version_{args.version}\n")
        for ck in ckpts:
            f.write(f"## \tcheckpoint: {ck}\n")
        f.write(f"#"*20+"\n")
        f.write(df_scores.to_csv(sep = "\t", float_format="%.3f", index=True))
    print(df_scores)



