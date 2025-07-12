import os
import yaml
import torch
from torch.optim import Adam,AdamW,lr_scheduler

from datasets.dataset import load_datasets, prepare_datasets
from models.fedOAP import UNetWithCrossAttention
from models.fedDP import UNetWithAttention
from train import train,test

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

  with open(os.path.join('conf/fedOAP.yaml'), "r") as f:
    cfg = yaml.safe_load(f)
  
  datasets = load_datasets(dataset_dirs={'lung':'datasets/lung_tumor'},image_size=cfg['image_size'])
  dataloaders = prepare_datasets(
    datasets=datasets, 
    batch_size=cfg['batch_size'], 
    num_clients=cfg['num_clients'], 
    random_seed=cfg['random_seed'], 
    train_ratio=cfg['train_ratio'], 
    val_ratio=cfg['val_ratio']
  )
  train_dataloader = dataloaders[0][0]
  val_dataloader = dataloaders[1][0]
  test_dataloader = dataloaders[2][0]

  # model_server_path = os.path.join(cfg['output_dir'],'fedOAPserver.pth')
  model_server_path = os.path.join('outputs/fedDP','fedDPserver.pth')

  if 'fedDP' in model_server_path:
    model = UNetWithAttention(
      in_channels = cfg['input_channels'],
      num_classes = cfg['num_classes']
    )
  else:
    model = UNetWithCrossAttention(
      in_channels = cfg['input_channels'],
      num_classes = cfg['num_classes']
    )
  
  if os.path.exists(model_server_path):
    model.load_state_dict(
      torch.load(model_server_path)
    )
  else:
    print('FedOAP Server is not available')
    exit()
  
  loss,iou,dice = test(
    model=model,
    test_dataloader=test_dataloader,
    device=DEVICE
  )
  print(f'FedOAP has {dice} dice score on Unseen Lung tumor Data')


  optimizer = AdamW(
    model.parameters(), 
    lr=cfg['config_fit']['lr'], 
    weight_decay=cfg['config_fit']['weight_decay']
  )
  scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=10, 
    T_mult=2, 
    eta_min=cfg['config_fit']['min_lr']
  )

  history = train(
    model = model,
    train_dataloader = train_dataloader, 
    optimizer = optimizer, 
    scheduler = scheduler, 
    epochs = 2, 
    device = DEVICE
  )

  loss,iou,dice = test(
    model=model,
    test_dataloader=test_dataloader,
    device=DEVICE
  )
  print(f'FedOAP has {dice} dice score on Unseen Lung tumor Data just after 2 epochs of finetuning')