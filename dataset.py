import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import random
import pyBigWig


size=2**11*5 # 10240
num_channel=6
batch_size=100
dna_path = ''

if_time=False
max_scale=1.15
min_scale=1
if_mag=False
max_mag=1.15
min_mag=0.9
if_flip=False


chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
num_bp=np.array([249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,155270560])

chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

list_dna=['A','C','G','T']
dict_dna={}
for the_id in list_dna:
    dict_dna[the_id]=pyBigWig.open(dna_path + the_id + '.bigwig')

def get_index(chr_set):
    tmp=[]
    for the_chr in chr_set:
        tmp.append(chr_len[the_chr])
    freq=np.rint(np.array(tmp)/sum(tmp)*1000).astype('int')
    index_set1=np.array([])
    for i in np.arange(len(chr_set)):
        index_set = np.hstack((index_set1, np.array([chr_set[i]] * freq[i])))
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

class UNetDataset(Dataset):
    def __init__(self, index_set, feature_bw, label_bw, is_train):
        self.index_set = index_set
        self.feature_bw = feature_bw
        self.label_bw = label_bw
        self.is_train = is_train

    def __len__(self):
        return len(self.index_set)

    def __getitem__(self, idx):
        the_chr = self.index_set[idx]

        start = int(np.random.randint(0, chr_len[the_chr] - size, 1))
        end = start + size

        label = np.array(self.label_bw.values(the_chr, start, end))

        image = np.zeros((num_channel, size)) # 6*10240
        num = 0
        for k in np.arange(len(list_dna)):
            the_id = list_dna[k]
            image[num, :] = dict_dna[the_id].values(the_chr, start, end)
            num += 1
        image[num, :] = np.array(self.feature_bw.values(the_chr, start, end))
        avg = np.array(self.feature_avg.values(the_chr, start, end))
        image[num + 1, :] = image[num, :] - avg

        if self.is_train and if_mag:
            rrr = random.random()
            rrr_mag = rrr * (max_mag - min_mag) + min_mag
            image[4, :] = image[4, :] * rrr_mag

        return torch.tensor(image.T).float(), torch.tensor(label.T).float()

class UNetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, par, feature_train, label_train, feature_vali, label_vali):
        super().__init__()
        self.batch_size = batch_size
        self.index_set1, self.index_set2 = get_index_set(par)
        self.feature_train = feature_train
        self.label_train = label_train
        self.feature_vali = feature_vali
        self.label_vali = label_vali

    def setup(self, stage=None):
        # Create datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = UNetDataset(self.index_set1, self.feature_train, self.label_train, is_train=True)
            self.val_dataset = UNetDataset(self.index_set2, self.feature_vali, self.label_vali, is_train=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

# Usage
# data_module = CustomDataModule(batch_size=32)
# data_module.setup('fit')
# train_loader = data_module.train_dataloader()
# val_loader = data_module.val_dataloader()
