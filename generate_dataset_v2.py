#!/usr/bin/env python

import os, sys 
import h5py
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import pyBigWig
import argparse
from tqdm import tqdm
import numpy as np
# import cupy as cp
import pandas as pd
import multiprocessing as mp
import numba
from collections import namedtuple


def get_args():
    opt = argparse.ArgumentParser(description='generate dataset')
    opt.add_argument("-a","--average", required=True, type=str, help = "path of average dnase bigwig")
    opt.add_argument("-b","--feature", required=True, type=str, help = "path of feature dnase bigwig")
    opt.add_argument("--blacklist", required=True, type=str, help = "path of blacklist")
    opt.add_argument("--ignore", required=False, type=str, default=None, help = "path of relaxed region")
    opt.add_argument("-l","--label", required=True, type=str, help = "path of label bed (e.g narrowPeak)")
    opt.add_argument("--chrom", required = True, default=[], action='append', help = "chromosome to use")
    
    opt.add_argument("--bin", required=False, type=int, default=1, help = "bin size")
    opt.add_argument("-n","--num", required=False, type=int, default=10000, help = "number of samples")
    opt.add_argument("-r","--region", required=False, default = "random", help="region to use, random or given regions in bed file (no header, 4 columns: chrom, start, end, index)")
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
dict_chromsize = dict(zip(List_chroms, List_chrom_sizes))

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
        yield chrom, start, end, i

def generate_regions(regions,df_chromsize = None, chroms =[], num = 10000, size = 10240, seed = 1024):
    if regions == "random":
        return random_region(df_chromsize, chroms, num, size, seed)
    else:
        assert os.path.exists(regions), f"region file not found at {regions}"
        df = pd.read_csv(regions, sep = "\t", names = ["chrom","start","end","ref_index"]).query("chrom in @chroms")
        center = (df['start'] + df['end'])//2
        df['start'] = center - size//2
        df['end'] = center + size//2
        df = df[df['start'] >= 0]
        df = df.sample(frac=1, random_state = seed).reset_index(drop=True)
        for i,row in enumerate(df.itertuples()):
            if i >= num:
                break
            yield row.chrom, row.start, row.end, row.ref_index
        return None
        
def get_total_num(region, chroms, num):
    if region == "random":
        return num
    else:
        assert os.path.exists(region), f"region file not found at {region}"
        df = pd.read_csv(region, sep = "\t", names = ["chrom","start","end","ref_index"]).query("chrom in @chroms")
        return min(num, len(df))



def bin_value(bin_arr):
    if np.sum(bin_arr == 1) >= 100:
        return 1
    elif np.sum(bin_arr == -1) >= 100:
        return -1
    else:
        return 0


def init_labels(peak,chroms):
    df_peak = pd.read_csv(peak, usecols=[0,1,2], sep = "\t", header = None)
    df_peak.columns = ["chrom","start","end"]
    df_peak = df_peak[df_peak['chrom'].isin(chroms)].sort_values(["chrom","start"]).reset_index(drop=True)
    dict_labels = {chrom:np.zeros(dict_chromsize[chrom], dtype=np.int8) for chrom in chroms}
    for row in df_peak.itertuples():
        dict_labels[row.chrom][row.start:row.end] = 1
    return dict_labels

def init_bw_values(bw, chroms):
    bwf = pyBigWig.open(bw)
    dict_bw = {chrom:bwf.values(chrom,0,dict_chromsize[chrom], numpy = True) for chrom in chroms}
    for key, value in dict_bw.items():
        assert len(value) == dict_chromsize[key], f"length of {key} is not correct"
    bwf.close()
    return dict_bw

# @numba.jit(nopython=False)
def process(args):


    # bwf_ave = pyBigWig.open(args.average)
    # bwf_feature = pyBigWig.open(args.feature)
    arr_bw_ave = init_bw_values(args.average, args.chrom)
    arr_bw_feature = init_bw_values(args.feature, args.chrom)

    labels = np.zeros((args.num, int(args.window/args.bin)), dtype=np.int8)
    values_ave = np.zeros((args.num, args.window), dtype=np.float16)
    values_feature = np.zeros((args.num, args.window), dtype=np.float16)
    regions = np.zeros((args.num, 4), dtype = np.int32)


    total_num = get_total_num(args.region, args.chrom, args.num)
    
    regionos_generated = generate_regions(
        regions = args.region,
        df_chromsize = df_chromsize,
        chroms = args.chrom,
        num = args.num,
        size = args.window,
        seed = args.seed

    )
    for i, (chrom, start, end, index) in tqdm(enumerate(regionos_generated), total = total_num):
        if args.ignore:
            label_long = dict_ignore[chrom][start:end].copy() * -1
            label_long[dict_labels[chrom][start:end] == 1] = 1

        else:
            label_long = dict_labels[chrom][start:end]

        # ignore blacklist
        label_long[dict_black[chrom][start:end] == 1] = -1

        if args.bin > 1:
            binned_label = label_long.reshape((-1, args.bin))
            labels[i] = np.array([bin_value(bin_arr) for bin_arr in binned_label])
        else:
            labels[i] = label_long
        regions[i,:] = np.array([dict_chrom_to_id[chrom], start, end, index])
        # values_ave[i,:] = bwf_ave.values(chrom, start, end, numpy=True)
        # values_feature[i,:] = bwf_feature.values(chrom, start, end, numpy=True)
        values_ave[i,:] = arr_bw_ave[chrom][start:end]
        values_feature[i,:] = arr_bw_feature[chrom][start:end]


    # bwf_ave.close()
    # bwf_feature.close()
    return labels, values_ave, values_feature, regions

if __name__ == "__main__":
    args = valid_args(get_args())

    dict_labels = init_labels(args.label, args.chrom)
    dict_black = init_labels(args.blacklist, args.chrom)


    if args.ignore:
        dict_ignore = init_labels(args.ignore, args.chrom)
    else:
        dict_ignore = None

    labels, values_ave, values_feature, regions = process(args)

    with h5py.File(args.opath, "w") as f:
        f.create_dataset("labels", data = labels, dtype = np.int8, chunks = (32,int(args.window/args.bin)))
        f.create_dataset("values_ave", data = values_ave, dtype=np.float16, chunks = (32,args.window) )
        f.create_dataset("values_feature", data = values_feature, dtype=np.float16, chunks = (32,args.window))
        f.create_dataset("regions", data = np.array(regions), dtype = np.int32, chunks = (32,4))
