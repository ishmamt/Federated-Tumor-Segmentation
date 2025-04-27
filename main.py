import hydra
import flwr as fl
from flwr.common.logger import log
from logging import INFO
from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig
import os
import sys
import yaml
import glob
import pickle
import torch
import traceback
from torch.optim import AdamW,lr_scheduler
import argparse
from tqdm import tqdm

from datasets.dataset import prepare_datasets, load_datasets
from flower.client import generate_client_function
from flower.server import get_on_fit_config_function, get_eval_function
from metrics import iou_dice_score
from finetune import FinetuneFedOAP,FineTuneFedDP
from models.UNet import UNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def mainFedOAP(args,cfg):
  # Parse config file and print it out
  log(INFO,'The config that is followed for this run')
  log(INFO, cfg)

  # Check if CUDA is available and log it
  if torch.cuda.is_available():
      log(INFO,"Running on CUDA compatible GPU")
  else:
      log(INFO,"Running on CPU")
  
  # Load Datasets
  datasets = load_datasets(cfg['dataset_dirs'], cfg['image_size'])
  log(INFO, f"Datasets loaded. Number of datasets: {len(datasets)}")
  train_dataloaders, val_dataloaders, test_dataloaders = prepare_datasets(
    datasets=datasets, 
    batch_size=cfg['batch_size'], 
    num_clients=cfg['num_clients'], 
    random_seed=cfg['random_seed'], 
    train_ratio=cfg['train_ratio'], 
    val_ratio=cfg['val_ratio']
  )
  
  # Define Clients
  client_function = generate_client_function(
    strategy=args.strategy,
    train_dataloaders=train_dataloaders, 
    val_dataloaders=val_dataloaders, 
    input_channels=cfg['input_channels'], 
    num_classes=cfg['num_classes'], 
    output_dir=cfg['output_dir'],
    random_seed=cfg['random_seed']
  )
  
  ## Upto this checked

  #Define Strategy
  strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.0001, 
    min_fit_clients=cfg['num_clients_per_round_fit'], 
    fraction_evaluate=0.0001, 
    min_evaluate_clients=cfg['num_clients_per_round_eval'], 
    min_available_clients=cfg['num_clients'], 
    on_fit_config_fn=get_on_fit_config_function(cfg['config_fit']),
    evaluate_fn=get_eval_function(
      strategy=args.strategy,
      input_channels=cfg['input_channels'], 
      num_classes=cfg['num_classes'], 
      val_dataloaders=val_dataloaders,
      output_dir=cfg['output_dir'], 
      random_seed=cfg['random_seed']
    )
  )
  
  # Start Simulation
  try:
    history = fl.simulation.start_simulation(
        client_fn=client_function,
        num_clients=cfg['num_clients'],
        config=fl.server.ServerConfig(num_rounds=cfg['num_rounds']),
        strategy=strategy,
        client_resources={
            "num_cpus": 2,
            "num_gpus": 1
        }
    )
  except Exception as e:
    log(INFO, f"While simulating an error has occured : {e}")
    traceback.print_exc()
    temporaryWeightsPaths = glob.glob('temporaryWeights/*.pth')
    for path in temporaryWeightsPaths:
      os.remove(path)
    exit()
  finally:
    temporaryWeightsPaths = glob.glob('temporaryWeights/*.pth')
    for path in temporaryWeightsPaths:
      os.remove(path)

  try:
    log(INFO, "Going into Finetuning")
    trainer = FinetuneFedOAP(
      train_dataloaders=train_dataloaders,
      val_dataloaders=val_dataloaders,
      test_dataloaders=test_dataloaders,
      config_fit=cfg['config_fit'],
      device=device,
      output_dir=cfg['output_dir'],
      epochs=cfg['finetuning_epochs'],
      val_per_epoch=cfg['val_per_epoch'],
      in_channels=cfg['input_channels'],
      num_classes=cfg['num_classes']
    )

    trainer.train()
    trainer.test()
  except Exception as e:
    log(INFO, f"While finetuning an error has occured : {e}")
    traceback.print_exc()
    for idx in range(cfg['num_clients']):
      finetunedName = f'fedOAPfinetuned{idx}.pth'
      os.remove(os.path.join(cfg['output_dir'],finetunedName))
    os.remove(os.path.join(cfg['output_dir'],'results.txt'))

  del trainer



def mainFedDP(args,cfg):
  # Parse config file and print it out
  log(INFO,'The config that is followed for this run')
  log(INFO, cfg)

  # Check if CUDA is available and log it
  if torch.cuda.is_available():
      log(INFO,"Running on CUDA compatible GPU")
  else:
      log(INFO,"Running on CPU")
  
  # Load Datasets
  datasets = load_datasets(cfg['dataset_dirs'], cfg['image_size'])
  log(INFO, "Datasets loaded. Number of datasets: %s", len(datasets))
  train_dataloaders, val_dataloaders, test_dataloaders = prepare_datasets(
    datasets=datasets, 
    batch_size=cfg['batch_size'], 
    num_clients=cfg['num_clients'], 
    random_seed=cfg['random_seed'], 
    train_ratio=cfg['train_ratio'], 
    val_ratio=cfg['val_ratio']
  )
  
  # Define Clients
  client_function = generate_client_function(
    strategy=args.strategy,
    train_dataloaders=train_dataloaders, 
    val_dataloaders=val_dataloaders, 
    input_channels=cfg['input_channels'], 
    num_classes=cfg['num_classes'], 
    output_dir=cfg['output_dir'],
    random_seed=cfg['random_seed']
  )
  
  #Define Strategy
  strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.0001, 
    min_fit_clients=cfg['num_clients_per_round_fit'], 
    fraction_evaluate=0.0001, 
    min_evaluate_clients=cfg['num_clients_per_round_eval'], 
    min_available_clients=cfg['num_clients'], 
    on_fit_config_fn=get_on_fit_config_function(cfg['config_fit']),
    evaluate_fn=get_eval_function(
      strategy=args.strategy,
      input_channels=cfg['input_channels'], 
      num_classes=cfg['num_classes'], 
      val_dataloaders=val_dataloaders,
      output_dir=cfg['output_dir'], 
      random_seed=cfg['random_seed']
    )
  )
  
  # Start Simulation
  try:
    history = fl.simulation.start_simulation(
      client_fn=client_function,
      num_clients=cfg['num_clients'],
      config=fl.server.ServerConfig(num_rounds=cfg['num_rounds']),
      strategy=strategy,
      client_resources={
          "num_cpus": 2,
          "num_gpus": 1
      }
    )
  except Exception as e:
    print(f"While simulating an error has occured : {e}")
    traceback.print_exc()
    temporaryWeightsPaths = glob.glob('temporaryWeights/*.pth')
    for path in temporaryWeightsPaths:
      os.remove(path)
    exit()
  finally:
    queryWeightsPaths = glob.glob('temporaryWeights/*.pth')
    for weight_path in queryWeightsPaths:
      os.remove(weight_path)


  try:
    log(INFO, "Going into Finetuning")
    trainer = FineTuneFedDP(
      train_dataloaders=train_dataloaders,
      val_dataloaders=val_dataloaders,
      test_dataloaders=test_dataloaders,
      config_fit=cfg['config_fit'],
      device=device,
      output_dir=cfg['output_dir'],
      epochs=cfg['igc_epochs'],
      val_per_epoch=cfg['val_per_epoch'],
      in_channels=cfg['input_channels'],
      num_classes=cfg['num_classes']
    )

    trainer.train()
    trainer.test()
  except Exception as e:
    log(INFO, f"While finetuning an error has occured : {e}")
    traceback.print_exc()
    for idx in range(cfg['num_clients']):
      finetunedName = f'fedDPfinetuned{idx}.pth'
      os.remove(os.path.join(cfg['output_dir'],finetunedName))
    os.remove(os.path.join(cfg['output_dir'],'results.txt'))

  del trainer



def mainFedAVG(args,cfg):
  # Parse config file and print it out
  log(INFO,'The config that is followed for this run')
  log(INFO, cfg)

  # Check if CUDA is available and log it
  if torch.cuda.is_available():
      log(INFO,"Running on CUDA compatible GPU")
  else:
      log(INFO,"Running on CPU")
  
  # Load Datasets
  datasets = load_datasets(cfg['dataset_dirs'], cfg['image_size'])
  log(INFO, "Datasets loaded. Number of datasets: %s", len(datasets))
  train_dataloaders, val_dataloaders, test_dataloaders = prepare_datasets(
    datasets=datasets, 
    batch_size=cfg['batch_size'], 
    num_clients=cfg['num_clients'], 
    random_seed=cfg['random_seed'], 
    train_ratio=cfg['train_ratio'], 
    val_ratio=cfg['val_ratio']
  )
  
  # Define Clients
  client_function = generate_client_function(
    strategy=args.strategy,
    train_dataloaders=train_dataloaders, 
    val_dataloaders=val_dataloaders, 
    input_channels=cfg['input_channels'], 
    num_classes=cfg['num_classes'], 
    output_dir=cfg['output_dir'],
    random_seed=cfg['random_seed']
  )
    
  #Define Strategy
  strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.0001, 
    min_fit_clients=cfg['num_clients_per_round_fit'], 
    fraction_evaluate=0.0001, 
    min_evaluate_clients=cfg['num_clients_per_round_eval'], 
    min_available_clients=cfg['num_clients'], 
    on_fit_config_fn=get_on_fit_config_function(cfg['config_fit']),
    evaluate_fn=get_eval_function(
      strategy=args.strategy,
      input_channels=cfg['input_channels'], 
      num_classes=cfg['num_classes'], 
      val_dataloaders=val_dataloaders,
      output_dir=cfg['output_dir'], 
      random_seed=cfg['random_seed']
    )
  )
  
  # Start Simulation
  try:
    history = fl.simulation.start_simulation(
        client_fn=client_function,
        num_clients=cfg['num_clients'],
        config=fl.server.ServerConfig(num_rounds=cfg['num_rounds']),
        strategy=strategy,
        client_resources={
            "num_cpus": 2,
            "num_gpus": 1
        }
    )
  except Exception as e:
    print(f"While simulating an error has occured : {e}")
    traceback.print_exc()
    exit()
  
  results = []
  model = UNet(
    in_channels=cfg['input_channels'],
    num_classes=cfg['num_classes'],
    random_seed=cfg['random_seed']
  )
  for idx in range(cfg['num_clients']):
    model.eval()
    model.to(device)
    
    with torch.no_grad():
      loop = tqdm(test_dataloaders[idx])
      
      for images, masks in loop:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        iou, dice = iou_dice_score(outputs, masks)
        results.append(dice)
  
  with open(os.path.join(cfg['output_dir'],'results.txt'),'w') as f:
    for idx in range(cfg['num_clients']):
      if idx > 0 : f.write(f'\nclient {idx} has dice score :{results[idx]}')
      else : f.write(f'client {idx} has dice score {results[idx]}')
      
            
if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Example script with int, string, and float arguments")

  # parser.add_argument('--number', type=int, required=True, help='An integer value')
  parser.add_argument('--strategy', type=str, required=True, help='Defining strategy like fedAvg or fedDP')
  parser.add_argument('--conf-path',type=str, default='conf', help='Defining path for configs dir')
  # parser.add_argument('--score', type=float, required=True, help='A float value')

  args = parser.parse_args()

  print('The arguments followed for this run')
  print(args)
  print('-'*50)

  with open(os.path.join(args.conf_path,f"{args.strategy}.yaml"), "r") as f:
    cfg = yaml.safe_load(f)

  if args.strategy == 'fedOAP': 
    mainFedOAP(args,cfg)
  elif args.strategy == 'fedDP':
    mainFedDP(args,cfg)
  elif args.strategy == 'fedAVG':
    mainFedAVG(args,cfg)
  else :
    print('The given strategy is not yet implemented.')
