import torch

def iou_dice_score(output, target, epsilon=1e-5):
    """
    Calculates Intersection over Union (IoU) and Dice Score for binary segmentation.
    
    Arguments:
    output (Tensor): Output tensor (logits).
    target (Tensor): Target tensor (binary mask).
    epsilon (float): Smoothing factor for Dice score.
    
    Returns:
    iou (float): Intersection over Union score.
    dice (float): Dice Score.
    """
    
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
        
    output_ = output > 0.5
    target_ = target > 0.5
    
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + epsilon) / (union + epsilon)
    dice = (2 * iou) / (iou + 1)
    
    return iou, dice


def dice_coef(output, target, epsilon=1e-5):
    """
    Calculates Dice Coefficient for binary segmentation.
    
    Arguments:
    output (Tensor): Output tensor (logits).
    target (Tensor): Target tensor (binary mask).
    epsilon (float): Smoothing factor for Dice Coefficient.
    
    Returns:
    dice (float): Dice Coefficient.
    """
    
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    
    intersection = (output * target).sum()
    total = output.sum() + target.sum()

    return (2. * intersection + epsilon) / (total + epsilon)