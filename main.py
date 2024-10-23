import hydra
import flwr as fl
from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig
import os
import pickle

from datasets.dataset import prepare_dataset
from datasets.BUSI import BUSIDataset
from client import generate_client_function
from server import get_on_fit_config_function, get_eval_function


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # Parse config file and print it out
    print(OmegaConf.to_yaml(cfg))
    
    # Load Dataset
    dataset = BUSIDataset(cfg.dataset_dir, image_size=cfg.image_size)
    train_dataloaders, val_dataloaders, test_dataloader = prepare_dataset(dataset, cfg.batch_size, num_partitions=cfg.num_clients, 
                                                                          random_seed=cfg.random_seed, train_ratio=cfg.train_ratio, 
                                                                          val_ratio=cfg.val_ratio)
    
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
                                         evaluate_fn=get_eval_function(cfg.input_channels, cfg.num_classes, test_dataloader))
    
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