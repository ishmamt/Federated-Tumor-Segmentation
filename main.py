import hydra
import flwr as fl
from flwr.common.logger import log
from logging import INFO
from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig
import os
import pickle
import torch

from datasets.dataset import prepare_datasets, load_datasets
from datasets.BUSI import BUSIDataset
from flower.client import generate_client_function
from flower.server import get_on_fit_config_function, get_eval_function


@hydra.main(config_path="conf", config_name="base", version_base=None)
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
    history = fl.simulation.start_simulation(
        client_fn=client_function,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": 2
            # "num_gpus": 0.5
        }
    )
    
    # Save simulation results
    output_dir = HydraConfig.get().runtime.output_dir
    results_output_dir = os.path.join(output_dir, "results.pkl")
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
    main()