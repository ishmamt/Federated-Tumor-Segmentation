import torch
from tqdm import tqdm

from metrics import iou_score
from loss import BCEDiceLoss
from utils import AvgMeter


def train(model, train_loader, val_loader, optimizer, scheduler, epochs, device, logger):
    criterion = BCEDiceLoss()
    
    # model.train()
    # model.to(device)
    
    train_avg_meters = {"loss": AvgMeter(), "iou": AvgMeter(), "dice": AvgMeter()}
    
    best_dice = 0.0
    trigger = 0  # Threshold for early stopping
    
    for epoch in range(epochs):
        print(f"Epoch {epoch} / {epochs}:\n")
        loop = tqdm(train_loader)
        model.train()
        model.to(device)
        
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            iou, dice = iou_score(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_avg_meters["loss"].update(loss.item())
            train_avg_meters["iou"].update(iou)
            train_avg_meters["dice"].update(dice)
            
            loop.set_postfix({"IoU": iou, "Dice": dice, "loss": loss.item()})
        
        scheduler.step()
        
        trigger += 1
        val_mean_loss, val_mean_iou, val_mean_dice = test(model, val_loader, device)
        
        logger.update(epoch, scheduler.get_last_lr()[0], 
                      train_avg_meters["loss"].avg, train_avg_meters["iou"].avg, train_avg_meters["dice"].avg,
                      val_mean_loss, val_mean_iou, val_mean_dice)
        
        if val_mean_dice > best_dice:
            best_dice = val_mean_dice
            trigger = 0
        
        if trigger >= 40:
            print(f"Early stopping at epoch: {epoch}")
            break


def test(model, test_loader, device):
    criterion = BCEDiceLoss()
    test_avg_meters = {"loss": AvgMeter(), "iou": AvgMeter(), "dice": AvgMeter()}
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        loop = tqdm(test_loader)
        
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            iou, dice = iou_score(outputs, masks)
            loss = criterion(outputs, masks).item()
            
            test_avg_meters["loss"].update(loss)
            test_avg_meters["iou"].update(iou)
            test_avg_meters["dice"].update(dice)
            
            loop.set_postfix({"IoU": iou, "Dice": dice, "loss": loss})
    
    return test_avg_meters["loss"].avg, test_avg_meters["iou"].avg, test_avg_meters["dice"].avg