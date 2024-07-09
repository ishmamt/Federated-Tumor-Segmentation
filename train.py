import torch
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

from metrics import iou_score


def train(model, train_loader, optimizer, epochs, device):
    criterion = BCEWithLogitsLoss()
    model.train()
    model.to(device)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch} / {epochs}:\n")
        loop = tqdm(train_loader)
        
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            iou, dice = iou_score(outputs, masks)
            loss.backward()
            optimizer.step()
            
            loop.set_postfix({"IoU": iou, "Dice": dice, "loss": loss.item()})


def test(model, test_loader, device):
    criterion = BCEWithLogitsLoss()
    total_iou, total_dice, loss = 0.0, 0.0, 0.0
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        loop = tqdm(test_loader)
        
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            iou, dice = iou_score(outputs, masks)
            total_iou += iou
            total_dice += dice
            loss += criterion(outputs, masks).item()
            
            loop.set_postfix({"IoU": iou, "Dice": dice, "loss": loss.item()})
    
    mean_iou = total_iou / len(test_loader.dataset)
    mean_dice = total_dice / len(test_loader.dataset)
    
    return loss, mean_iou, mean_dice