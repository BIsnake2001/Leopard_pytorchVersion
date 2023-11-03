import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam

ss = 10

def crossentropy_cut(y_true, y_pred):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
    mask = y_true >= -0.5
    losses = -(y_true * torch.log(y_pred) + (1.0 - y_true) * torch.log(1.0 - y_pred))
    losses = losses[mask]
    masked_loss = losses.mean()
    return masked_loss

def dice_coef(y_true, y_pred):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    mask = y_true >= -0.5
    intersection = torch.sum(y_true * y_pred * mask)
    return (2. * intersection + ss) / (torch.sum(y_true * mask) + torch.sum(y_pred * mask) + ss)
 
class LitUNetFT(pl.LightningModule):

    def __init__(self, unet_model):
        super().__init__()
        self.unet_model = unet_model

    def forward(self, data):
        return self.unet_model.forward(data)


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = crossentropy_cut(y, y_pred)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = crossentropy_cut(y, y_pred)
        dice_score = dice_coef(y, y_pred)
        self.log('val_loss', loss)
        self.log('val_dice_coef', dice_score)
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-5)
        return optimizer

