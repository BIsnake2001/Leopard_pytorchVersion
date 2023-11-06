import re
import torch
import random
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
import numpy as np
from pyfaidx import Fasta


size=2**11*5 # 10240
num_channel=6
batch_size=100
dna_path = '/shared/zhangyuxuan/projects/Model/scripts/1.finetune/18.Leopard/data/hg38'

if_time=False
max_scale=1.15
min_scale=1
if_mag=False
max_mag=1.15
min_mag=0.9
if_flip=False


chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
num_bp=np.array([248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973, 145138636, 138394717, 133797422, 135086622, 133275309, 114364328, 107043718, 101991189, 90338345, 83257441, 80373285, 58617616, 64444167, 46709983, 50818468, 156040895])

chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

chr_index = {}
for i in np.arange(len(chr_all)):
    chr_index[chr_all[i]]=i

def get_index(chr_set):
    tmp=[]
    for the_chr in chr_set:
        tmp.append(chr_len[the_chr])
    freq=np.rint(np.array(tmp)/sum(tmp)*100000).astype('int')
    index_set=np.array([])
    for i in np.arange(len(chr_set)):
        index_set = np.hstack((index_set, np.array([chr_set[i]] * freq[i])))
    print(len(index_set))
    return index_set

def get_index_set(par):
    chr_train_all=['chr2','chr3','chr4','chr5','chr6','chr7','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr22','chrX']
    ratio=0.5
    np.random.seed(int(par))
    np.random.shuffle(chr_train_all)
    tmp=int(len(chr_train_all)*ratio)
    chr_set1=chr_train_all[:tmp]
    chr_set2=chr_train_all[tmp:]
    index_set1 = get_index(chr_set1)
    index_set2 = get_index(chr_set2)
    return index_set1, index_set2

def seq_to_hot(seq):
    encoding = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}

    seq = seq.upper()

    # Replace non-standard bases with 'N'
    seq = re.sub('[BD-FH-SU-Z]', 'N', seq)

    hot = np.array([encoding.get(base, [0, 0, 0, 0]) for base in seq], dtype='int').T

    return hot

class UNetDataset(Dataset):
    def __init__(self, index_set, feature_bw, label_array, feature_avg, is_train):
        self.index_set = index_set
        self.feature_bw = feature_bw
        self.label_array = label_array
        self.is_train = is_train
        self.feature_avg = feature_avg
        self.fasta = Fasta('/shared/zhangyuxuan/data/annotation/hg38.fa')

    def __len__(self):
        return len(self.index_set)

    def __getitem__(self, idx):
        the_chr = self.index_set[idx]
        the_row = chr_index[the_chr]

        start = int(np.random.randint(0, chr_len[the_chr] - size, 1))
        end = start + size

        label = self.label_array[the_row, start:end]

        image = np.zeros((num_channel, size)) # 6*10240
        seq = self.fasta[the_chr][start:end].seq
        image[:4, :] = seq_to_hot(seq)
        num = 4
        image[num, :] = np.array(self.feature_bw.values(the_chr, start, end))
        avg = np.array(self.feature_avg.values(the_chr, start, end))
        image[num + 1, :] = image[num, :] - avg

        if self.is_train and if_mag:
            rrr = random.random()
            rrr_mag = rrr * (max_mag - min_mag) + min_mag
            image[4, :] = image[4, :] * rrr_mag

        return torch.tensor(image).float(), torch.tensor(label).float()

class UNetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, par, feature_train, label_train, feature_vali, label_vali, feature_avg):
        super().__init__()
        self.batch_size = batch_size
        self.index_set1, self.index_set2 = get_index_set(par)
        self.feature_train = feature_train
        self.label_train = label_train
        self.feature_vali = feature_vali
        self.feature_avg = feature_avg
        self.label_vali = label_vali

    def setup(self, stage=None):
        # Create datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = UNetDataset(self.index_set1, self.feature_train, self.label_train, self.feature_avg, is_train=True)
            self.val_dataset = UNetDataset(self.index_set2, self.feature_vali, self.label_vali, self.feature_avg, is_train=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

# Usage
# data_module = CustomDataModule(batch_size=32)
# data_module.setup('fit')
# train_loader = data_module.train_dataloader()
# val_loader = data_module.val_dataloader()
