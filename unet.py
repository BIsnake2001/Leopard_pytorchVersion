import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import AdamW
from unet_model import UNet
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score, average_precision_score

ss = 10

def crossentropy_cut(y_true, y_prob):
    y_true = y_true.view(-1)
    y_prob = y_prob.view(-1)
    y_prob = torch.clamp(y_prob, 1e-7, 1 - 1e-7)
    mask = y_true >= -0.5
    losses = -(y_true * torch.log(y_prob) + (1.0 - y_true) * torch.log(1.0 - y_prob))
    losses = losses[mask]
    masked_loss = losses.mean()
    return masked_loss

def dice_coef(y_true, y_prob):
    y_true = y_true.view(-1)
    y_prob= y_prob.view(-1)
    mask = y_true >= -0.5
    intersection = torch.sum(y_true * y_prob * mask)
    return (2. * intersection + ss) / (torch.sum(y_true * mask) + torch.sum(y_prob * mask) + ss)
 
class LitUNetFT(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.unet_model = UNet()
        self.lr = 1e-3

    def forward(self, data):
        return self.unet_model.forward(data)


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_prob = self.forward(x)
        loss = crossentropy_cut(y, y_prob)
        self.log('train_loss', loss, on_step = True, on_epoch = True, prog_bar = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_prob= self.forward(x)
        loss = crossentropy_cut(y, y_prob)
        dice_score = dice_coef(y, y_prob)

        self.log('val_loss', loss, on_step = True, on_epoch = True, prog_bar = True)
        self.log('val_dice_coef', dice_score, on_step = True, on_epoch = True, prog_bar = True)
        y = y.view(-1)
        y_prob = y_prob.view(-1)
        y_true = y.cpu().numpy()
        y_pred = (y_prob > 0.5).cpu().numpy()
        nonmask = y_true != -1
        y_true = y_true[nonmask]
        y_pred = y_pred[nonmask]
        
        f1 = f1_score(y_true, y_pred, average = 'binary')
        mcc = matthews_corrcoef(y_true, y_pred)
        pre = precision_score(y_true, y_pred, zero_division = np.nan)
        recall = recall_score(y_true, y_pred, zero_division = np.nan)
        if (not np.isnan(pre)) or (not np.isnan(recall)):
            self.log('valid_F1', f1, on_step = True, on_epoch = True, prog_bar = True)
            self.log('valid_MCC', mcc, on_step = True, on_epoch = True, prog_bar = True)
        if not np.isnan(pre):
            self.log('valid_precision', pre, on_step = True, on_epoch = True, prog_bar = True)
        if not np.isnan(recall):
            self.log('valid_recall', recall, on_step = True, on_epoch = True, prog_bar = True)
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-5)

        lr_scheduler = {
        'scheduler': MultiStepLR(optimizer, milestones=[5], gamma=0.1),
        'name': 'my_scheduler',  # optional
        'interval': 'epoch',  # or 'step'
        'frequency': 1,  # optional
        'reduce_on_plateau': False,  # For ReduceLROnPlateau scheduler
        'monitor': 'val_loss',  # optional
        'strict': True,  # optional
    }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

