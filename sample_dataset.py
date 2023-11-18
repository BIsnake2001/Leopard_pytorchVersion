import os,sys 
import numpy as np
import pandas as pd
import h5py
import argparse
import pathlib

'''
The script will sample the regions to given fraction of input dataset
'''

def get_args():
    parser = argparse.ArgumentParser(description='sample dataset')
    parser.add_argument("-i","--input", required=True, type=str, help = "input dataset")
    parser.add_argument("-o","--output", required=True, type=str, help = "output dataset")

    parser.add_argument("-n","--num", required=False, type=int, default=1, help = "number of positive samples. if < 1, it will be treated as fraction. if > 1, it will be treated as number. if greater than max number of samples, it will be set to max number of samples")
    parser.add_argument("--ratio", default=0.1, type=float, help = "positive samples : negative samples")
    parser.add_argument("--seed", required=False, type=int, default=1024, help = "random seed")

    args = parser.parse_args()
    return args

def valid_args(args):
    assert os.path.exists(args.input), "input dataset not found"
    assert args.num > 0, "number of samples should be positive"

    oname = args.output
    if not oname.endswith(".h5"):
        oname += ".h5"
    oname = pathlib.Path(oname)
    oname.parent.mkdir(parents=True, exist_ok=True)
    args.oname = oname

    return args



def get_samples(dataset, ignore_index = -1, thre = 100):
    with h5py.File(dataset, "r") as f:
        labels = np.array(f["labels"][:,:])

    labels[labels == ignore_index] = 0
    l = labels.sum(axis = 1 ) > thre
    df_samples = pd.DataFrame({"label":l})
    df_samples["index"] = df_samples.index
    assert len(set(df_samples["label"])) == 2, "labels should be binary"
    return df_samples

def get_sample_num(df_samples, num, ratio):
    num_pos = df_samples.loc[df_samples["label"] == 1].shape[0]
    num_neg = df_samples.loc[df_samples["label"] == 0].shape[0]

    if num <= 1:
        num_pos_sampled = int(num_pos * num)
    else:
        num_pos_sampled = min(num, num_pos)

    num_neg_sampled = int(num_pos_sampled / ratio)
    num_neg_sampled = min(num_neg_sampled, num_neg)

    return num_pos_sampled, num_neg_sampled

    

    

    

def get_sample_index(df_samples, num = 1, seed = 1024):
    assert "index" in df_samples.columns, "index not found"
    assert "label" in df_samples.columns, "label not found"

    num_pos, num_neg = get_sample_num(df_samples, num, ratio = args.ratio)

    print(f"number of positive samples: {num_pos}")
    print(f"number of negative samples: {num_neg}")

    np.random.seed(seed)
    s1, s2 = np.random.randint(0, 10000, 2)

    df1 = df_samples.loc[df_samples["label"] == 1].sample(n = num_pos, random_state = s1)
    df2 = df_samples.loc[df_samples["label"] == 0].sample(n = num_neg, random_state = s2)

    df_index = pd.concat([df1, df2], axis = 0)

    return df_index["index"].values
    

def dump_by_index(dataset, index, oname, tables = ["labels", "values_ave", "values_feature", "regions"]):
    with h5py.File(dataset, "r") as f:
        with h5py.File(oname, "w") as g:
            for table in f.keys():
                if table not in tables:
                    g.create_dataset(table, data = f[table][:])
                else:
                    g.create_dataset(table, data = np.array(f[table][:,:])[index,:])
            g.create_dataset("index", data = index)
    return oname

if __name__ == '__main__':
    args = valid_args(get_args())
    print("# get samples")
    df_samples = get_samples(args.input)
    print("# sample index")
    index = get_sample_index(df_samples, args.num, seed = args.seed)
    print("# dump")
    dump_by_index(args.input, index, args.oname)
    print("# done")






