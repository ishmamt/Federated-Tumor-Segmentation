import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self):
        """
        Loss function for segmentation task combining Binary Cross Entropy and Dice Coefficient.
        """
        
        super().__init__()

    def forward(self, input, target, epsilon=1e-5):
        """
        Calculates the Binary Cross Entropy and Dice Coefficient loss function to the input tensors.
        
        Arguments:
        input (Tensor): Input tensor.
        target (Tensor): Target tensor.
        epsilon (float): Smoothing factor for Dice Coefficient.
        
        Returns:
        loss (Tensor): Loss value.
        """
        
        bce_loss = F.binary_cross_entropy_with_logits(input, target)
        
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + epsilon) / (input.sum(1) + target.sum(1) + epsilon)
        dice = 1 - dice.sum() / num
        
        return 0.5 * bce_loss + dice