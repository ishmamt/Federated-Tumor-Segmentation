import torch
from tqdm import tqdm

from metrics import iou_dice_score
from loss import BCEDiceLoss
from utils import AvgMeter



def trainWithAttention(model_tuple, train_dataloader, optimizer, scheduler, epochs, device):
    """
    Basic training loop for the client.
    
    Arguments:
    model (nn.Module): The model to be trained.
    train_dataloader (DataLoader): DataLoader for training data.
    optimizer (optimizer): Pytorch optimizer for training.
    scheduler (lr_scheduler): Learning rate scheduler for training.
    epochs (int): Number of training epochs.
    device (torch.device): Device for training.
    
    Returns:
    history (dict): Training history containing epochs, learning_rate, train_loss, train_iou, train_dice.
    """
    
    criterion = BCEDiceLoss()
    modelDown, qGenerator, modelUp = model_tuple
    modelDown.train()
    qGenerator.train()
    modelUp.train()
    modelDown.to(device)
    qGenerator.to(device)
    modelUp.to(device)
    
    train_avg_meters = {"loss": AvgMeter(), "iou": AvgMeter(), "dice": AvgMeter()}
    history = {"epoch": [], 
                "lr": [],
                "train_loss": [],
                "train_iou": [],
                "train_dice": []
                }
    
    for epoch in range(epochs):
        print(f"Epoch {epoch} / {epochs}:\n")
        loop = tqdm(train_dataloader)
        
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputsForUp = modelDown(images)
            qForUp = qGenerator(outputsForUp[0])
            outputs = modelUp(qForUp,outputsForUp)
            loss = criterion(outputs, masks)
            iou, dice = iou_dice_score(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_avg_meters["loss"].update(loss.item())
            train_avg_meters["iou"].update(iou)
            train_avg_meters["dice"].update(dice)
            
            loop.set_postfix({"IoU": iou, "Dice": dice, "loss": loss.item()})
        
        scheduler.step()
        
        # Updating history
        history["epoch"].append(epoch)
        history["lr"].append(scheduler.get_last_lr()[0])
        history["train_loss"].append(train_avg_meters["loss"].avg)
        history["train_iou"].append(train_avg_meters["iou"].avg)
        history["train_dice"].append(train_avg_meters["dice"].avg)
        
    return history


def testWithAttention(model_tuple, test_dataloader, device):
    """
    Basic testing loop for the client.
    
    Arguments:
    model (nn.Module): The model to be tested.
    test_dataloader (DataLoader): DataLoader for testing data.
    device (torch.device): Device for testing.
    
    Returns:
    loss (float): Average loss over the test set.
    iou (float): Average IoU score over the test set.
    dice (float): Average Dice score over the test set.
    """
    
    criterion = BCEDiceLoss()
    test_avg_meters = {"loss": AvgMeter(), "iou": AvgMeter(), "dice": AvgMeter()}
    modelDown, qGenerator, modelUp = model_tuple
    modelDown.eval()
    qGenerator.eval()
    modelUp.eval()
    modelDown.to(device)
    qGenerator.to(device)
    modelUp.to(device)
    
    with torch.no_grad():
        loop = tqdm(test_dataloader)
        
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            outputsForUp = modelDown(images)
            qForUp = qGenerator(outputsForUp[0])
            outputs = modelUp(qForUp,outputsForUp)
            
            iou, dice = iou_dice_score(outputs, masks)
            loss = criterion(outputs, masks).item()
            
            test_avg_meters["loss"].update(loss)
            test_avg_meters["iou"].update(iou)
            test_avg_meters["dice"].update(dice)
            
            loop.set_postfix({"IoU": iou, "Dice": dice, "loss": loss})
    
    return test_avg_meters["loss"].avg, test_avg_meters["iou"].avg, test_avg_meters["dice"].avg



def trainWithAdapter(model, adapter, train_dataloader, optimizer, scheduler, epochs, device):
    """
    Basic training loop for the client.
    
    Arguments:
    model (nn.Module): The model to be trained.
    train_dataloader (DataLoader): DataLoader for training data.
    optimizer (optimizer): Pytorch optimizer for training.
    scheduler (lr_scheduler): Learning rate scheduler for training.
    epochs (int): Number of training epochs.
    device (torch.device): Device for training.
    
    Returns:
    history (dict): Training history containing epochs, learning_rate, train_loss, train_iou, train_dice.
    """
    
    criterion = BCEDiceLoss()
    model.train()
    adapter.train()
    model.to(device)
    adapter.to(device)
    
    train_avg_meters = {"loss": AvgMeter(), "iou": AvgMeter(), "dice": AvgMeter()}
    history = {"epoch": [], 
                "lr": [],
                "train_loss": [],
                "train_iou": [],
                "train_dice": []
                }
    
    for epoch in range(epochs):
        print(f"Epoch {epoch} / {epochs}:\n")
        loop = tqdm(train_dataloader)
        
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputsForAdaptation = model(images)
            outputs = adapter(outputsForAdaptation)
            loss = criterion(outputs, masks)
            iou, dice = iou_dice_score(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_avg_meters["loss"].update(loss.item())
            train_avg_meters["iou"].update(iou)
            train_avg_meters["dice"].update(dice)
            
            loop.set_postfix({"IoU": iou, "Dice": dice, "loss": loss.item()})
        
        scheduler.step()
        
        # Updating history
        history["epoch"].append(epoch)
        history["lr"].append(scheduler.get_last_lr()[0])
        history["train_loss"].append(train_avg_meters["loss"].avg)
        history["train_iou"].append(train_avg_meters["iou"].avg)
        history["train_dice"].append(train_avg_meters["dice"].avg)
        
    return history


def testWithAdapter(model, adapter, test_dataloader, device):
    """
    Basic testing loop for the client.
    
    Arguments:
    model (nn.Module): The model to be tested.
    test_dataloader (DataLoader): DataLoader for testing data.
    device (torch.device): Device for testing.
    
    Returns:
    loss (float): Average loss over the test set.
    iou (float): Average IoU score over the test set.
    dice (float): Average Dice score over the test set.
    """
    
    criterion = BCEDiceLoss()
    test_avg_meters = {"loss": AvgMeter(), "iou": AvgMeter(), "dice": AvgMeter()}
    model.eval()
    adapter.eval()
    model.to(device)
    adapter.to(device)
    
    with torch.no_grad():
        loop = tqdm(test_dataloader)
        
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            outputsForAdaptation = model(images)
            outputs = adapter(outputsForAdaptation)
            
            iou, dice = iou_dice_score(outputs, masks)
            loss = criterion(outputs, masks).item()
            
            test_avg_meters["loss"].update(loss)
            test_avg_meters["iou"].update(iou)
            test_avg_meters["dice"].update(dice)
            
            loop.set_postfix({"IoU": iou, "Dice": dice, "loss": loss})
    
    return test_avg_meters["loss"].avg, test_avg_meters["iou"].avg, test_avg_meters["dice"].avg