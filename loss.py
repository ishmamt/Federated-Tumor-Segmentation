import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, epsilon=1e-5):
        bce_loss = F.binary_cross_entropy_with_logits(input, target)
        
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + epsilon) / (input.sum(1) + target.sum(1) + epsilon)
        dice = 1 - dice.sum() / num
        
        return 0.5 * bce_loss + dice