import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.optim import lr_scheduler

from datasets.BUSI import BUSIDataset
from train import train, test
from UNet import UNet
from utils import show_image, Logger


if __name__ == '__main__':
    LR = 0.001
    MIN_LR = 0.00001
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 16
    EPOCHS = 200
    IMAGE_SIZE = 128
    IN_CHANNELS = 1
    NUM_CLASSES = 1
    ROOT_DIR = os.path.join("data", "Dataset_BUSI_with_GT")
    OUTPUT_DIR = os.path.join("outputs")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEBUG_SPLIT = 0.6
    
    logger = Logger(OUTPUT_DIR)
    
    dataset = BUSIDataset(ROOT_DIR, image_size=IMAGE_SIZE)
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [DEBUG_SPLIT, 1 - DEBUG_SPLIT], generator=generator)
    test_dataset, val_dataset = random_split(val_dataset, [0.5, 0.5], generator=generator)
    
    # print(train_dataset)
    # train_dataset = train_dataset[0]
    # val_dataset = train_dataset
    print(len(train_dataset))
    # print(train_dataset)
    # exit()
    
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset, 
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    test_dataloader = DataLoader(test_dataset, 
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    
    model = UNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)
    
    train(model, train_dataloader, val_dataloader, optimizer, scheduler, EPOCHS, DEVICE, logger)
    loss, mean_iou, mean_dice = test(model, test_dataloader, DEVICE)
    
    print("-" * 50)
    print(f"Test loss: {loss}")
    print(f"Test IoU: {mean_iou}")
    print(f"Test Dice: {mean_dice}")
    print("-" * 50)
    
    model.eval()
    model.to(DEVICE)
    with torch.no_grad():
        for images, masks in test_dataloader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            
            output = outputs[0].detach().cpu().numpy()
            mask = masks[0].detach().cpu().numpy()
            show_image(mask, output)