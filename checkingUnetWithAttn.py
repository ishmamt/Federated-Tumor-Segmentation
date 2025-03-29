import torch
import torch.optim as optim

from datasets.dataset import prepare_datasets, load_datasets
from models.unetGpt import UNetWithAttention
from train import train,test

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
  model = UNetWithAttention(in_channels=1, num_classes=1)
  optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
  scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=10, 
    T_mult=2, 
    eta_min=1e-6
  )
  # x = torch.randn(1, 3, 256, 256)  # Sample input
  # y = model(x)
  # print(y.shape) 

  datasets = load_datasets(
    {'colon' : "/content/drive/MyDrive/UFF/CVC-ColonDB"}, 
    128
  )

  # print(len(datasets))
  
  train_dataloaders, val_dataloaders, test_dataloaders = \
  prepare_datasets(
    datasets = datasets, 
    batch_size = 16, 
    num_clients=1, 
    random_seed=42, 
    train_ratio=0.9, 
    val_ratio=0.1
  )
  
  # print(f"size of train_dataloaders {len(train_dataloaders)}")
  # exit()

  history = train(
    model = model,
    train_dataloader = train_dataloaders[-1],
    optimizer = optimizer,
    scheduler = scheduler,
    epochs = 200,
    device = device
  )

  loss,iou,dice = test(
    model = model, 
    test_dataloader = test_dataloaders[-1],
    device = device
  )

  print(f"test loss is {loss}")
  print(f"test iou is {iou}")
  print(f"test dice is {dice}")