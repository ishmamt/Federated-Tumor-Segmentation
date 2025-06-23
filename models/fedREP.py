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


class ModelSplit(ABC, nn.Module):
    """Abstract class for splitting a model into body and head."""

    def __init__(self, model: nn.Module):
        """Initialize the attributes of the model split.

        Args:
            model: dict containing the vocab sizes of the input attributes.
        """
        super().__init__()

        self._body, self._head = self._get_model_parts(model)

    @abstractmethod
    def _get_model_parts(self, model: nn.Module) -> Tuple[nn.Module, nn.Module]:
        """Return the body and head of the model.

        Args:
            model: model to be split into head and body

        Returns
        -------
            Tuple where the first element is the body of the model
            and the second is the head.
        """

    @property
    def body(self) -> nn.Module:
        """Return model body."""
        return self._body

    @body.setter
    def body(self, state_dict: OrderedDict[str, Tensor]) -> None:
        """Set model body.

        Args:
            state_dict: dictionary of the state to set the model body to.
        """
        self._body.load_state_dict(state_dict, strict=True)

    @property
    def head(self) -> nn.Module:
        """Return model head."""
        return self._head

    @head.setter
    def head(self, state_dict: OrderedDict[str, Tensor]) -> None:
        """Set model head.

        Args:
            state_dict: dictionary of the state to set the model head to.
        """
        self._head.load_state_dict(state_dict, strict=True)

    def get_parameters(self) -> NDArrays:
        """Get model parameters.

        Returns
        -------
            Body and head parameters
        """
        return [
            val.cpu().numpy()
            for val in [
                *self.body.state_dict().values(),
                *self.head.state_dict().values(),
            ]
        ]

    def set_parameters(self, state_dict: Dict[str, Tensor]) -> None:
        """Set model parameters.

        Args:
            state_dict: dictionary of the state to set the model to.
        """
        self.load_state_dict(state_dict, strict=False)

    def enable_head(self) -> None:
        """Enable gradient tracking for the head parameters."""
        for param in self._head.parameters():
            param.requires_grad = True

    def enable_body(self) -> None:
        """Enable gradient tracking for the body parameters."""
        for param in self._body.parameters():
            param.requires_grad = True

    def disable_head(self) -> None:
        """Disable gradient tracking for the head parameters."""
        for param in self._head.parameters():
            param.requires_grad = False

    def disable_body(self) -> None:
        """Disable gradient tracking for the body parameters."""
        for param in self._body.parameters():
            param.requires_grad = False

    def forward(self, inputs: Any) -> Any:
        """Forward inputs through the body and the head."""
        return self.head(self.body(inputs))


# pylint: disable=R0902, R0913, R0801
class ModelManager(ABC):
    """Manager for models with Body/Head split."""

    def __init__(
        self,
        client_id: int,
        cfg: Dict,
        trainloader: DataLoader,
        testloader: DataLoader,
        in_channels: int = 1,
        num_classes: int = 1,
        model_split_class=ModelSplit,  # ModelSplit
    ):
        """Initialize the attributes of the model manager.

        Args:
            cfg: The config of the current run.
            trainloader: Client train dataloader.
            testloader: Client test dataloader.
            model_split_class: Class to be used to split the model into body and head \
                (concrete implementation of ModelSplit).
        """
        super().__init__()
        self.client_id = client_id,
        self.cfg = cfg
        self.trainloader = trainloader
        self.testloader = testloader
        self.in_channels = in_channels
        self.num_classes = num_classes
        self._model: ModelSplit = model_split_class(self._create_model())
        

    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Return model to be split into head and body."""

    @property
    def model(self) -> ModelSplit:
        """Return model."""
        return self._model

    def _load_client_state(self) -> None:
        if os.path.exists(self.headWeightsPath):
            self._model.head.load_state_dict(torch.load(self.headWeightsPath))
        else:
            print('head parameter is not found')


    def _save_client_state(self) -> None:
        print('Saving head weights')
        torch.save(self._model.head.state_dict(), self.headWeightsPath)

    def train(self, device: torch.device):
        self._load_client_state()

        num_local_epochs = self.cfg['local_epochs']

        num_rep_epochs = self.cfg['rep_epochs']

        criterion = BCEDiceLoss()
        optimizer = AdamW(
            self._model.parameters(), 
            lr = self.cfg['lr'], 
            weight_decay = self.cfg['weight_decay']
        )
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10, 
            T_mult=2, 
            eta_min=self.cfg['min_lr']
        )

        iou, dice, total_itr = 0.0, 0.0, 0
        loss_avg = 0.0

        self._model.to(device)
        self._model.train()

        history = {"epoch": [], 
            "lr": [],
            "train_loss": [],
            "train_iou": [],
            "train_dice": []
        }

        for i in range(num_local_epochs + num_rep_epochs):
            if i < num_local_epochs:
                self._model.disable_body()
                self._model.enable_head()
            else:
                self._model.enable_body()
                self._model.disable_head()

            loop = tqdm(self.trainloader)
            for images,masks in loop:

                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                outputs = self._model(images)
                loss = criterion(outputs, masks)
                iu, dc = iou_dice_score(outputs, masks)
                loss.backward()
                optimizer.step()

                total_itr += 1
                iou += iu
                dice += dc
                loss_avg += loss.item()

                loop.set_postfix({"IoU": iu, "Dice": dc, "loss": loss.item()})

            scheduler.step()

            history["epoch"].append(i)
            history["lr"].append(scheduler.get_last_lr()[0])
            history["train_loss"].append(loss_avg/total_itr)
            history["train_iou"].append(iou/total_itr)
            history["train_dice"].append(dice/total_itr)

        # Save state.
        self._save_client_state()

        return history

    def test(self, device: torch.device):
        
        self._load_client_state()

        num_finetune_epochs = self.cfg['finetune_epochs']

        if num_finetune_epochs > 0 :
            
            criterion = BCEDiceLoss()
            optimizer = AdamW(
                self._model.parameters(), 
                lr = self.cfg['lr'], 
                weight_decay = self.cfg['weight_decay']
            )
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=10, 
                T_mult=2, 
                eta_min=self.cfg['min_lr']
            )

            iou, dice, total_itr = 0.0, 0.0, 0
            loss_avg = 0.0

            self._model.to(device)
            self._model.train()

            for _ in range(num_finetune_epochs):
                loop = tqdm(self.trainloader)
                for images,masks in loop:
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = self._model(images)
                    loss = criterion(outputs, masks)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        criterion = BCEDiceLoss()
        iou, dice, total_itr, loss_avg = 0.0, 0.0, 0, 0.0

        self._model.to(device)
        self._model.eval()
        with torch.no_grad():

            loop = tqdm(self.testloader)

            for images,masks in loop:
                images = images.to(device)
                masks = masks.to(device)
                outputs = self._model(images)
                loss = criterion(outputs, masks)
                total_itr += 1
                iu, dc = iou_dice_score(outputs, masks)
                iuo += iu
                dice += dc
                loss_avg += loss.item()
                loop.set_postfix({"IoU": iu, "Dice": dc, "loss": loss.item()})
                

        return loss_avg, iou, dice

    def train_dataset_size(self) -> int:
        """Return train data set size."""
        return len(self.trainloader.dataset)

    def test_dataset_size(self) -> int:
        """Return test data set size."""
        return len(self.testloader.dataset)

    def total_dataset_size(self) -> int:
        """Return total data set size."""
        return len(self.trainloader.dataset) + len(self.testloader.dataset)



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

class UnetFedRep(nn.Module):
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

class UnetModelSplit(ModelSplit):
    def _get_model_parts(
            self, model:UnetFedRep
        ) -> Tuple[nn.Module, nn.Module]:

        return model.body, model.head

class UnetModelManager(ModelManager):
    """Manager for models with Body/Head split."""

    def __init__(self, **kwargs):
        """Initialize the attributes of the model manager."""
        super().__init__(model_split_class=UnetModelSplit, **kwargs)

    def _create_model(self) -> nn.Module:
        """Return Unet model to be split into head and body."""
        return UnetFedRep(in_channels=self.in_channels,num_classes=self.num_classes)