import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import AdamW
from unet_model import UNet
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score, average_precision_score

import torchmetrics as tm

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

class BCEWithLogitsLossIgnore(nn.Module):
    def __init__(self, ignore_index = -1):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss = nn.BCEWithLogitsLoss(reduction = 'none')
    
    def forward(self, y_logit, y_true):
        mask = y_true != self.ignore_index
        loss = self.loss(y_logit, y_true)
        loss = loss[mask]
        masked_loss = loss.mean()
        return masked_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, y_true, y_prob):
        y_true = y_true.view(-1)
        y_prob = y_prob.view(-1)
        # y_prob = torch.clamp(y_prob, 1e-7, 1 - 1e-7)
        mask = y_true >= -0.5
        bce = -(y_true * torch.log(y_prob) + (1.0 - y_true) * torch.log(1.0 - y_prob))
        loss = self.alpha * (1.0 - y_prob) ** self.gamma * bce
        loss = loss[mask]
        masked_loss = loss.mean()
        return masked_loss

class LogTensorValues:
    def __init__(self):
        self.log_data = {}

    def log(self, key, values):
        if key not in self.log_data:
            self.log_data[key] = []
        self.log_data[key].append(values.reshape(-1).squeeze())

    def get_values(self, key):
        if key not in self.log_data:
            return None
        # return np.concatenate(self.log_data[key])
        return torch.cat(self.log_data[key])


class LitUNetFT(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.unet_model = UNet()
        self.lr = 1e-3
        self.loss_function = BCEWithLogitsLossIgnore(ignore_index = -1)
        self.metric_precision = tm.Precision(task="binary")
        self.metric_recall = tm.Recall(task="binary")
        self.metric_mcc = tm.MatthewsCorrCoef(task="binary")
        self.metric_f1 = tm.F1Score(task="binary")
        self.metric_auroc = tm.AUROC(task="binary")
        self.metric_auprc = tm.AveragePrecision(task="binary")
        self.metric_acc = tm.Accuracy(task="binary")


    def forward(self, data):
        return self.unet_model.forward(data)


    def training_step(self, batch, batch_idx):
        x, y = batch["inputs"], batch["label"]
        y_logit = self.forward(x)
        # loss = crossentropy_cut(y, y_prob)
        loss = self.loss_function(y_logit.view(-1), y.view(-1))
        self.log('train_loss', loss, on_step = True, on_epoch = True, prog_bar = True)
        return loss
    
    def on_validation_epoch_start(self):
        self.logger_values = LogTensorValues()

    def validation_step(self, batch, batch_idx):
        x, y = batch["inputs"], batch["label"]
        y_logit= self.forward(x)
        y_prob = torch.sigmoid(y_logit)
        loss = self.loss_function(y_logit.view(-1), y.view(-1))
        dice_score = dice_coef(y, y_prob)

        self.log('val_loss', loss, on_step = True, on_epoch = True, prog_bar = True)
        self.log('val_dice_coef', dice_score, on_step = True, on_epoch = True, prog_bar = True)
        y = y.view(-1)
        y_prob = y_prob.view(-1)
        y_true = y
        y_pred = (y_prob > 0.5)
        nonmask = y_true != -1

        self.logger_values.log('y_true', y_true[nonmask])
        self.logger_values.log('y_pred', y_pred[nonmask])
        self.logger_values.log('y_prob', y_prob[nonmask])
        # y_true = y_true[nonmask]
        # y_pred = y_pred[nonmask]
        

        # pre = precision_score(y_true, y_pred, zero_division = np.nan)
        # recall = recall_score(y_true, y_pred, zero_division = np.nan)

        # if not np.isnan(pre):
        #     self.log('valid_precision', pre, on_step = False, on_epoch = True, prog_bar = True)
        # if not np.isnan(recall):
        #     self.log('valid_recall', recall, on_step = False, on_epoch = True, prog_bar = True)

        # if (not np.isnan(pre)) or (not np.isnan(recall)):
        #     f1 = f1_score(y_true, y_pred, average = 'binary')
        #     mcc = matthews_corrcoef(y_true, y_pred)
        #     self.log('valid_F1', f1, on_step = False, on_epoch = True, prog_bar = True)
        #     self.log('valid_MCC', mcc, on_step = False, on_epoch = True, prog_bar = True)
        return loss
            
    def on_validation_epoch_end(self):
        y_true = self.logger_values.get_values('y_true')
        y_pred = self.logger_values.get_values('y_pred')
        y_prob = self.logger_values.get_values('y_prob')



        self.log("valid_label_pos_frac", y_true.mean(), sync_dist = False, on_step = False, on_epoch = True, prog_bar = True, batch_size = y_true.shape[0])
        self.log("valid_pred_pos_frac", y_pred.float().mean(), sync_dist = False, on_step = False, on_epoch = True, prog_bar = True, batch_size = y_true.shape[0])
        self.log("valid_pred_prob_mean", y_prob.mean(), sync_dist = False, on_step = False, on_epoch = True, prog_bar = True, batch_size = y_true.shape[0])

        acc = self.metric_acc(y_prob,y_true)
        self.log('valid_acc', acc, sync_dist = False, on_step = False, on_epoch = True, prog_bar = True, batch_size = y_true.shape[0])

        if y_pred.sum() > 0:
            mr = self.metric_recall(y_pred,y_true)
            self.log('valid_recall', mr, sync_dist = False, on_step = False, on_epoch = True, prog_bar = True, batch_size = y_true.shape[0])
        if y_pred.sum() > 0 or y_true.sum() > 0:
            self.log('valid_MCC', self.metric_mcc(y_pred,y_true), sync_dist = False, on_step = False, on_epoch = True, prog_bar = True, batch_size = y_true.shape[0])
            self.log('valid_F1', self.metric_f1(y_pred,y_true), sync_dist = False, on_step = False, on_epoch = True, prog_bar = True, batch_size = y_true.shape[0])
        if y_true.sum() > 0:
            mp = self.metric_precision(y_prob,y_true)
            self.log('valid_precision', mp, sync_dist = False, on_step = False, on_epoch = True, prog_bar = True, batch_size = y_true.shape[0])

            auroc = self.metric_auroc(y_prob, y_true)
            self.log('valid_AUROC', auroc, sync_dist = False, on_step = False, on_epoch = True, prog_bar = True, batch_size = y_true.shape[0])
            auprc = self.metric_auprc(y_prob, y_true)
            self.log('valid_AUPRC', auprc, sync_dist = False, on_step = False, on_epoch = True, prog_bar = True, batch_size = y_true.shape[0])
        self.logger_values = None
        # self.trainer.datamodule.setup('val')
        return None

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

