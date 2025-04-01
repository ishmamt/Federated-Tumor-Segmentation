import hydra
import flwr as fl
from flwr.common.logger import log
from logging import INFO
from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig
import os
import glob
import pickle
import torch
import traceback
from copy import deepcopy
from torch.optim import AdamW,lr_scheduler

from datasets.dataset import prepare_datasets, load_datasets
from flower.clientWithModifications import generate_client_function
from flower.serverWithModifications import get_on_fit_config_function, get_eval_function
from models.unetGpt import UNetWithAttention
from trainWithModifications import train,test

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def performingIGC(serverWeightsPath, cfg, train_dataloaders, test_dataloaders):

  clients = []

  for idx in range(cfg.num_clients):
    queryWeightsPath = f'/content/drive/MyDrive/UFF/Federated-Tumor-Segmentation/q_weight/query{idx}.pth'
    model = UNetWithAttention(
      in_channels = cfg.input_channels,
      num_classes = cfg.num_classes
    )
    model.load_state_dict(torch.load(serverWeightsPath)) ## Loading Server's Weight
    model.attn.query.load_state_dict(torch.load(queryWeightsPath)) ## Loading Query's Weights

    clients.append(model.to(device))

  clientsForIgc = deepcopy(clients)

  losses,ious,dices = [],[],[]

  for idx in range(cfg.num_clients):
    optimizer = AdamW(clients[idx].parameters(), lr=cfg.config_fit['lr'], weight_decay=cfg.config_fit['weight_decay'])
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
      optimizer, 
      T_0=10, 
      T_mult=2, 
      eta_min=cfg.config_fit['min_lr']
    )

    history = train(
      model = clients[idx],
      train_dataloader=train_dataloaders[idx],
      optimizer = optimizer,
      scheduler = scheduler,
      epochs = cfg.igc_epochs,
      device = device,
      method = 'igc',
      clients = clientsForIgc
    )

    loss,iou,dice = test(
      model = clients[idx],
      test_dataloader = test_dataloaders[idx],
      device = device
    )
    losses.append(loss)
    ious.append(iou)
    dices.append(dice)

  with open('/content/drive/MyDrive/UFF/Federated-Tumor-Segmentation/outputs_without_FL/fedDp/resultLQ&IGC.txt','w') as f:
    for idx in range(cfg.num_clients):
      f.write(f'loss of client {idx} is {losses[idx]}\n')
      f.write(f'iou of client {idx} is {ious[idx]}\n')
      f.write(f'dice of client {idx} is {dices[idx]}\n')
      f.write('\n')



@hydra.main(config_path="conf", config_name="without_FL", version_base=None)
def mainWithAttention(cfg:DictConfig):
  # Parse config file and print it out
  print(OmegaConf.to_yaml(cfg))
  
  # Check if CUDA is available and log it
  if torch.cuda.is_available():
      log(INFO, "Running on CUDA compatible GPU")
  else:
      log(INFO, "Running on CPU")
  
  # Load Datasets
  datasets = load_datasets(cfg.dataset_dirs, cfg.image_size)
  log(INFO, "Datasets loaded. Number of datasets: %s", len(datasets))
  train_dataloaders, val_dataloaders, test_dataloaders = prepare_datasets(datasets, cfg.batch_size, num_clients=cfg.num_clients, 
                                                                        random_seed=cfg.random_seed, train_ratio=cfg.train_ratio, 
                                                                        val_ratio=cfg.val_ratio)
  # exit()
  
  # Define Clients
  client_function = generate_client_function(train_dataloaders, val_dataloaders, cfg.input_channels, 
                                              cfg.num_classes, cfg.random_seed)
  
  #Define Strategy
  strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.0001, 
    min_fit_clients=cfg.num_clients_per_round_fit, 
    fraction_evaluate=0.0001, 
    min_evaluate_clients=cfg.num_clients_per_round_eval, 
    min_available_clients=cfg.num_clients, 
    on_fit_config_fn=get_on_fit_config_function(cfg.config_fit),
    evaluate_fn=get_eval_function(
      cfg.input_channels, 
      cfg.num_classes, 
      test_dataloaders, 
      cfg.random_seed
    )
  )
  
  # Start Simulation
  try:
    history = fl.simulation.start_simulation(
        client_fn=client_function,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": 2,
            "num_gpus": 1
        }
    )
  except Exception as e:
    print(f"While simulating an error has occured : {e}")
    traceback.print_exc()
    queryWeightsPaths = glob.glob('/content/drive/MyDrive/UFF/Federated-Tumor-Segmentation/q_weight/*.pth')
    for path in queryWeightsPaths:
      os.remove(path)
    os.remove('/content/drive/MyDrive/UFF/Federated-Tumor-Segmentation/outputs_without_FL/fedDp/unetWithLQ.pth')
    exit()
  finally:
    pass
    # queryWeightsPaths = glob.glob('queryWeights/*.pth')
    # for weight_path in queryWeightsPaths:
    #   os.remove(weight_path)
    # exit()
  
  # Save simulation results
  output_dir = HydraConfig.get().runtime.output_dir
  # output_dir = cfg.output_dir
  results_output_dir = os.path.join(output_dir, "results.pkl")
  # print(f'results_output_dir : {results_output_dir}')
  # exit()
  results = {"history": history}
  
  with open(results_output_dir, 'wb') as f:
      pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

  try:
    performingIGC(
      serverWeightsPath = '/content/drive/MyDrive/UFF/Federated-Tumor-Segmentation/outputs_without_FL/fedDp/unetWithLQ.pth',
      cfg = cfg,
      train_dataloaders = train_dataloaders,
      test_dataloaders = test_dataloaders
    )
  except Exception as e:
    print(f"While simulating an error has occured : {e}")
    traceback.print_exc()
    queryWeightsPaths = glob.glob('/content/drive/MyDrive/UFF/Federated-Tumor-Segmentation/q_weight/*.pth')
    for path in queryWeightsPaths:
      os.remove(path)
    os.remove('/content/drive/MyDrive/UFF/Federated-Tumor-Segmentation/outputs_without_FL/fedDp/unetWithLQ.pth')
    exit()

  queryWeightsPaths = glob.glob('/content/drive/MyDrive/UFF/Federated-Tumor-Segmentation/q_weight/*.pth')
  for path in queryWeightsPaths:
      os.remove(path)
  os.remove('/content/drive/MyDrive/UFF/Federated-Tumor-Segmentation/outputs_without_FL/fedDp/unetWithLQ.pth')



@hydra.main(config_path="conf", config_name="without_FL", version_base=None)
def main(cfg: DictConfig):
    # Parse config file and print it out
    print(OmegaConf.to_yaml(cfg))
    
    # Check if CUDA is available and log it
    if torch.cuda.is_available():
        log(INFO, "Running on CUDA compatible GPU")
    else:
        log(INFO, "Running on CPU")
    
    # Load Datasets
    datasets = load_datasets(cfg.dataset_dirs, cfg.image_size)
    log(INFO, "Datasets loaded. Number of datasets: %s", len(datasets))
    train_dataloaders, val_dataloaders, test_dataloaders = prepare_datasets(datasets, cfg.batch_size, num_clients=cfg.num_clients, 
                                                                          random_seed=cfg.random_seed, train_ratio=cfg.train_ratio, 
                                                                          val_ratio=cfg.val_ratio)
    # exit()
    
    # Define Clients
    client_function = generate_client_function(train_dataloaders, val_dataloaders, cfg.input_channels, 
                                               cfg.num_classes, cfg.random_seed)
    
    #Define Strategy
    strategy = fl.server.strategy.FedAvg(fraction_fit=0.0001, 
                                         min_fit_clients=cfg.num_clients_per_round_fit, 
                                         fraction_evaluate=0.0001, 
                                         min_evaluate_clients=cfg.num_clients_per_round_eval, 
                                         min_available_clients=cfg.num_clients, 
                                         on_fit_config_fn=get_on_fit_config_function(cfg.config_fit),
                                         evaluate_fn=get_eval_function(cfg.input_channels, cfg.num_classes, test_dataloaders, 
                                                                       cfg.random_seed))
    
    # Start Simulation
    try:
      history = fl.simulation.start_simulation(
          client_fn=client_function,
          num_clients=cfg.num_clients,
          config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
          strategy=strategy,
          client_resources={
              "num_cpus": 2,
              "num_gpus": 1
          }
      )
    except Exception as e:
      print(f"Inside mainWithModifications While simulating an error has occured : {e}")
      traceback.print_exc()
      exit()
    finally:
      adapter_weight_paths = glob.glob('/content/drive/MyDrive/UFF/Federated-Tumor-Segmentation/adapter_weight/*.pth')
      for weight_path in adapter_weight_paths:
        os.remove(weight_path)
      # exit()
    
    # Save simulation results
    output_dir = HydraConfig.get().runtime.output_dir
    # output_dir = cfg.output_dir
    results_output_dir = os.path.join(output_dir, "results.pkl")
    # print(f'results_output_dir : {results_output_dir}')
    # exit()
    results = {"history": history}
    
    with open(results_output_dir, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    

    
    # model.eval()
    # model.to(DEVICE)
    # with torch.no_grad():
    #     for images, masks in test_dataloader:
    #         images, masks = images.to(DEVICE), masks.to(DEVICE)
    #         outputs = model(images)
            
    #         output = outputs[0].detach().cpu().numpy()
    #         mask = masks[0].detach().cpu().numpy()
    #         show_image(mask, output)
      
            
if __name__ == "__main__":

  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # modelDown = SharedDown(
  #   in_channels = 1,
  #   random_seed = 42
  # ).to(device)
  # print(modelDown.output_channels)
  # qGenerator = QGenerator(
  #   in_channels = modelDown.output_channels,
  #   random_seed=42
  # ).to(device)
  # print(qGenerator.in_channels,qGenerator.attn_dim,qGenerator.kernel_size)
  # modelUp = SharedUpWithAttn(
  #   in_channels = modelDown.output_channels,
  #   num_classes = 1,
  #   random_seed = 42
  # ).to(device)

  # images = torch.rand(1,1,128,128).to(device)

  # outputsForUp = modelDown(images)
  # qForUp = qGenerator(outputsForUp[0])
  # outputs = modelUp(qForUp,outputsForUp)
  # print(outputs.size())

  mainWithAttention()
  # main()