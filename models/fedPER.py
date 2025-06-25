"""fedrep: A Flower Baseline."""

import os
import collections
from abc import ABC, abstractmethod
from typing import Any, Dict, List, OrderedDict, Tuple, Union
from tqdm import tqdm
import numpy as np

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from flwr.common import NDArrays

from metrics import iou_dice_score
from loss import BCEDiceLoss

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Creates a Double Convolutional Block with a series of two convolutional layers and ReLU activation.
        
        Arguments:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        """
        
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Applies the Double Convolutional Block to the input tensor.
        
        Arguments:
        x (Tensor): Input tensor.
        
        Returns:
        output (Tensor): Output tensor after applying the Double Convolutional Block.
        """
        
        return self.conv_op(x)

# changing init function so that I can access out_channels
# by extending UNet class
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Creates a Downsampling Block with a series of convolutional layer and max pooling.
        
        Arguments:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        """
        
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.out_channels = out_channels

    def forward(self, x):
        """
        Applies the Downsampling Block to the input tensor.
        
        Arguments:
        x (Tensor): Input tensor.
        
        Returns:
        down (Tensor): Output tensor after downsampling.
        p (Tensor): Output tensor after max pooling.
        """
        
        down = self.conv(x)
        p = self.pool(down)

        return down, p


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Creates an Upsampling Block with a series of convolutional layers and upsampling.
        
        Arguments:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        """
        
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Applies the Upsampling Block to the input tensors.
        
        Arguments:
        x1 (Tensor): Input tensor after downsampling.
        x2 (Tensor): Input tensor before downsampling.
        
        Returns:
        x (Tensor): Output tensor after upsampling and concatenating.
        """
        
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        
        return self.conv(x)
   
class UnetBody(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)
    
    def forward(self, x):

        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)
        
        b = self.bottle_neck(p4)

        return (b, down_4, down_3, down_2, down_1)

class UnetHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)
        
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
    
    def forward(self,x):

        up_1 = self.up_convolution_1(x[0], x[1])
        up_2 = self.up_convolution_2(up_1, x[2])
        up_3 = self.up_convolution_3(up_2, x[3])
        up_4 = self.up_convolution_4(up_3, x[4])
        
        output = self.out(up_4)

        return output

class UnetFedPer(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        self.body = UnetBody(in_channels=in_channels)
        self.head = UnetHead(num_classes=num_classes)
    
    def enable_head(self) -> None:
        """Enable gradient tracking for the head parameters."""
        for param in self.head.parameters():
            param.requires_grad = True

    def enable_body(self) -> None:
        """Enable gradient tracking for the body parameters."""
        for param in self.body.parameters():
            param.requires_grad = True

    def disable_head(self) -> None:
        """Disable gradient tracking for the head parameters."""
        for param in self.head.parameters():
            param.requires_grad = False

    def disable_body(self) -> None:
        """Disable gradient tracking for the body parameters."""
        for param in self.body.parameters():
            param.requires_grad = False

    def forward(self,x):
        return self.head(self.body(x))