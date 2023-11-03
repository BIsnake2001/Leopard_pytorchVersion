import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Conv1DTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding='same'):
        super(Conv1DTranspose, self).__init__()
        self.conv2d_transpose = nn.ConvTranspose2d(in_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(padding, 0))

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.conv2d_transpose(x)
        x = x.squeeze(2)
        return x

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation='relu', padding='same'):
        super(UNetBlock, self).__init__()
        padding = kernel_size // 2 if padding == 'same' else 0
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.batch_norm1 = nn.BatchNorm1d(out_channels)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x))) if self.activation == 'relu' else self.batch_norm1(self.conv1(x))
        x = F.relu(self.batch_norm2(self.conv2(x))) if self.activation == 'relu' else self.batch_norm2(self.conv2(x))
        return x

class PCC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation='relu', padding='same'):
        super(PCC, self).__init__()
        self.pool = nn.MaxPool1d(2)
        self.unet_block = UNetBlock(in_channels, out_channels, kernel_size, activation, padding)

    def forward(self, x):
        x = self.pool(x)
        x = self.unet_block(x)
        return x

class UCC(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, kernel_size, activation='relu', padding='same'):
        super(UCC, self).__init__()
        self.conv_transpose = Conv1DTranspose(in_channels1, out_channels, 2)
        self.unet_block = UNetBlock(out_channels + in_channels2, out_channels, kernel_size, activation, padding)

    def forward(self, x1, x2):
        x1 = self.conv_transpose(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.unet_block(x)
        return x

class UNet(pl.LightningModule):
    def __init__(self, lr=1e-1, num_class=2, num_channel=10, size=2048*25):
        super(UNet, self).__init__()

        # Define the architecture here
        self.lr = lr
        num_blocks = 5
        initial_filter = 15
        scale_filter = 1.5
        size_kernel = 7
        activation = 'relu'
        padding = 'same'

        self.input_conv = UNetBlock(num_channel, initial_filter, size_kernel, activation, padding)

        self.down_blocks = nn.ModuleList()
        num_filters = initial_filter
        for i in range(num_blocks):
            num_filters = int(num_filters * scale_filter)
            self.down_blocks.append(PCC(num_filters // scale_filter, num_filters, size_kernel, activation, padding))

        self.up_blocks = nn.ModuleList()
        for i in range(num_blocks):
            num_filters = int(num_filters / scale_filter)
            self.up_blocks.append(UCC(num_filters * scale_filter, num_filters // scale_filter, num_filters, size_kernel, activation, padding))

        self.out_conv = nn.Conv1d(num_filters, num_class, 1)

    def forward(self, x):
        # Forward pass
        x = self.input_conv(x)

        down_outputs = [x]
        for block in self.down_blocks:
            x = block(x)
            down_outputs.append(x)

        for i, block in enumerate(self.up_blocks):
            x = block(x, down_outputs[-(i + 2)])

        x = torch.sigmoid(self.out_conv(x))
        return x

   