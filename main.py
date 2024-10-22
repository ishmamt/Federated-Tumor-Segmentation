import torch
import hydra
from torch.optim import Adam
from torch.optim import lr_scheduler
from omegaconf import OmegaConf, DictConfig

from datasets.dataset import prepare_dataset
from datasets.BUSI import BUSIDataset
from train import train, test
from UNet import UNet
from utils import show_image, Logger


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger = Logger(cfg.output_dir)
    
    # Parse config file and print it out
    print(OmegaConf.to_yaml(cfg))
    
    # Load Dataset
    dataset = BUSIDataset(cfg.dataset_dir, image_size=cfg.image_size)
    train_dataloaders, val_dataloaders, test_dataloader = prepare_dataset(dataset, cfg.batch_size, num_partitions=cfg.num_clients, 
                                                                          random_seed=cfg.random_seed, train_ratio=cfg.train_ratio, 
                                                                          val_ratio=cfg.val_ratio)
    
    exit()
    
    torch.manual_seed(cfg.random_seed)
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
      
            
if __name__ == "__main__":
    main()