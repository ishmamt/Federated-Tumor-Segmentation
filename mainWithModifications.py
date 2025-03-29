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

from datasets.dataset import prepare_datasets, load_datasets
# from datasets.BUSI import BUSIDataset
from flower.clientWithModifications import generate_client_function
from flower.serverWithModifications import get_on_fit_config_function, get_eval_function
# from flower.client import generate_client_function
# from flower.server import get_on_fit_config_function, get_eval_function
from models.unetWithModifications \
import SharedDown,QGenerator,SharedUpWithAttn

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
    print(f"While simulating an error has occured : {e}")
  finally:
    q_weight_paths = glob.glob('/content/drive/MyDrive/UFF/Federated-Tumor-Segmentation/q_weight/*.pth')
    for weight_path in q_weight_paths:
      os.remove(weight_path)
    exit()
  
  # Save simulation results
  output_dir = HydraConfig.get().runtime.output_dir
  # output_dir = cfg.output_dir
  results_output_dir = os.path.join(output_dir, "results.pkl")
  # print(f'results_output_dir : {results_output_dir}')
  # exit()
  results = {"history": history}
  
  with open(results_output_dir, 'wb') as f:
      pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)



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
      print(f"While simulating an error has occured : {e}")
    finally:
      adapter_weight_paths = glob.glob('/content/drive/MyDrive/UFF/Federated-Tumor-Segmentation/adapter_weight/*.pth')
      for weight_path in adapter_weight_paths:
        os.remove(weight_path)
      exit()
    
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