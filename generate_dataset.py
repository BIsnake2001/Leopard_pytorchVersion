#!/usr/bin/env python

import os, sys 
import h5py

import pyBigWig
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import multiprocessing as mp


def get_args():
    opt = argparse.ArgumentParser(description='generate dataset')
    opt.add_argument("-a","--average", required=True, type=str, help = "path of average dnase bigwig")
    opt.add_argument("-b","--feature", required=True, type=str, help = "path of feature dnase bigwig")
    opt.add_argument("--blacklist", required=True, type=str, help = "path of blacklist")
    opt.add_argument("--ignore", required=False,default = None, type=str, help = "path of relaxed region")
    opt.add_argument("-l","--label", required=True, type=str, help = "path of label bed (e.g narrowPeak)")
    opt.add_argument("--chrom", required = True, default=[], action='append', help = "chromosome to use")

    opt.add_argument("-n","--num", required=False, type=int, default=10000, help = "number of samples")
    opt.add_argument("-w","--window", required=False, type=int, default=1000, help = "window size")
    opt.add_argument("--seed", required=False, type=int, default=1, help = "random seed")
    opt.add_argument("-p", "--process", required=False, type=int, default=1, help = "number of processes")

    opt.add_argument("--odir", required=False, default=".", type=str, help = "output path")
    opt.add_argument("-o","--oname", required=False, default="dataset.h5", type=str, help = "output file name")


    return opt.parse_args()

def valid_args(args):
    print(args.average)
    assert os.path.exists(args.average), "average bigwig not found"
    assert os.path.exists(args.feature), "feature bigwig not found"
    assert os.path.exists(args.label), "label bed not found"

    chroms = [i.split(",") for i in args.chrom]
    chroms = [j for i in chroms for j in i if len(j) > 0]
    chroms = sorted(list(set(chroms)))
    args.chrom = chroms
    print (f"chromosomes to use: {args.chrom}")
    for chrom in args.chrom:
        assert chrom.startswith("chr"), "chromosome name should start with chr"
    assert len(args.chrom) > 0, "chromosome not specified"


    assert args.num > 0, "number of samples should be positive"
    assert args.window > 0, "window size should be positive"

    if not os.path.exists(args.odir):
        os.makedirs(args.odir)
    args.opath = os.path.join(args.odir, args.oname)

    return args

# get environment variables
List_chroms=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
List_chrom_sizes=np.array([248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973, 145138636, 138394717, 133797422, 135086622, 133275309, 114364328, 107043718, 101991189, 90338345, 83257441, 80373285, 58617616, 64444167, 46709983, 50818468, 156040895])
df_chromsize = pd.DataFrame({'chrom':List_chroms,'size':List_chrom_sizes}).assign(
    fraction= lambda x: x['size']/np.sum(x['size'])
)

dict_chrom_to_id = {f"chr{i}": i for i in range(1,23)}
dict_chrom_to_id['chrX'] = 24
dict_chrom_to_id['chrY'] = 25
    

def random_region(df_chromsize, chroms, num = 10000, size = 10240, seed = 1024):
    probs = df_chromsize[df_chromsize['chrom'].isin(chroms)]['fraction'].values
    probs = probs/np.sum(probs)
    np.random.seed(seed)
    for i in range(num):
        chrom = np.random.choice(chroms, p = probs)
        start = np.random.randint(0, df_chromsize[df_chromsize['chrom'] == chrom]['size'].values[0] - size - 10)
        end = start + size
        yield chrom, start, end


def process_region(chrom, start, end, df_peak_g, df_blc_g, df_ign_g, bwf_ave, bwf_feature, labels, values_ave, values_feature, regions, dict_chrom_to_id, i):

    if df_ign_g is not None:
        df_ign_region = df_ign_g.get_group(chrom).assign(
            start = lambda x: np.clip(x['start'] - start,0,args.window),
            end = lambda x: np.clip(x['end'] - start,0,args.window)
        ).query("start < end")
        for _, row in df_ign_region.iterrows():
            labels[i, row['start']:row['end']] = -1

    df_region = df_peak_g.get_group(chrom).assign(
        start = lambda x: np.clip(x['start'] - start,0,args.window),
        end = lambda x: np.clip(x['end'] - start,0,args.window)
    ).query("start < end")
    for _, row in df_region.iterrows():
        labels[i, row['start']:row['end']] = 1
        
    df_blc_region = df_blc_g.get_group(chrom).assign(
        start = lambda x: np.clip(x['start'] - start,0,args.window),
        end = lambda x: np.clip(x['end'] - start,0,args.window)
    ).query("start < end")
    
    for _, row in df_blc_region.iterrows():
        labels[i, row['start']:row['end']] = -1
    
    region_ = np.array([dict_chrom_to_id[chrom], start, end])
    regions[i,:] = region_

    value_ = bwf_ave.values(chrom, start, end)
    values_ave[i,:] = value_

    value_feature_ = bwf_feature.values(chrom, start, end)
    values_feature[i,:] = value_feature_
    return None


if __name__ == "__main__":
    args = valid_args(get_args())

    df_peak = pd.read_csv(args.label, usecols=[0,1,2], sep = "\t", header = None)
    df_peak.columns = ["chrom","start","end"]
    df_peak_g = df_peak.sort_values(["chrom","start","end"]).groupby("chrom")

    df_blc = pd.read_csv(args.blacklist, sep = '\t', header=None, names=['chrom', 'start', 'end'])
    df_blc_g = df_blc.sort_values(["chrom","start","end"]).groupby("chrom")

    if args.ignore is not None:
        df_ign = pd.read_csv(args.ignore, sep = '\t', header=None, names=['chrom', 'start', 'end'])
        df_ign_g = df_ign.sort_values(["chrom","start","end"]).groupby("chrom")
    else:
        df_ign_g = None

    bwf_ave = pyBigWig.open(args.average)
    bwf_feature = pyBigWig.open(args.feature)

    labels = np.zeros((args.num, args.window))
    values_ave = np.zeros((args.num, args.window))
    values_feature = np.zeros((args.num, args.window))
    regions = np.zeros((args.num, 3), dtype = np.int32)

    # pool = mp.Pool(args.process)

    for i, (chrom, start, end) in tqdm(enumerate(random_region(df_chromsize, args.chrom, args.num, args.window, args.seed)), total = args.num):
        # pool.apply_async(process_region, args=(chrom, start, end, df_peak_g, bwf_ave, bwf_feature, labels, values_ave, values_feature, regions, dict_chrom_to_id, i))
        process_region(chrom, start, end, df_peak_g, df_blc_g, df_ign_g, bwf_ave, bwf_feature, labels, values_ave, values_feature, regions, dict_chrom_to_id, i)



    # pool.close()
    # pool.join()

    bwf_ave.close()
    bwf_feature.close()

    with h5py.File(args.opath, "w") as f:
        f.create_dataset("labels", data = labels, dtype = np.int8, chunks = (32,args.window))
        f.create_dataset("values_ave", data = values_ave, dtype=np.float32, chunks = (32,args.window) )
        f.create_dataset("values_feature", data = values_feature, dtype=np.float32, chunks = (32,args.window))
        f.create_dataset("regions", data = np.array(regions), dtype = np.int32, chunks = (32,3))
