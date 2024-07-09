import torch

def iou_score(output, target, epsilon=1e-5):
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
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    
    intersection = (output * target).sum()
    total = output.sum() + target.sum()

    return (2. * intersection + epsilon) / (total + epsilon)