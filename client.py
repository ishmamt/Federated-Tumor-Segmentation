import flwr as fl
import torch
from collections import OrderedDict

from models.UNet import UNet


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, train_dataloader, val_dataloader, input_channels, num_classes, random_seed=42):
        """
        Creates a FlowerClient object to simulate a client.
        
        Arguments:
        train_dataloader (DataLoader): DataLoader for training data on a single client.
        val_dataloader (DataLoader): DataLoader for validation data on a single client.
        num_classes (int): Number of classes in the dataset.
        random_seed (int): Random seed for reproducibility.
        """
        
        super().__init__()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        torch.manual_seed(random_seed)
        self.model = UNet(in_channels=input_channels, num_classes=num_classes)
        
    def set_parameters(self, params):
        """
        Updates the model parameters using the given numpy arrays.
        
        Arguments:
        params (list): List of numpy arrays representing the model parameters.
        """
        
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        
    def get_parameters(self, config):
        """
        Returns the current model parameters as numpy arrays.
        
        Arguments:
        config (dict): Configuration dictionary.
        
        Returns:
        params (list): List of numpy arrays representing the model parameters.
        """
        
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    
def generate_client_function(train_dataloaders, val_dataloaders, input_channels, num_classes, random_seed):
    """
    Provides a client function that the server can evoke to spawn clients.
    
    Arguments:
    train_dataloaders (list): List of DataLoader objects for training data on multiple clients.
    val_dataloaders (list): List of DataLoader objects for validation data on multiple clients.
    input_channels (int): Number of input channels in the dataset.
    num_classes (int): Number of classes in the dataset.
    random_seed (int): Random seed for reproducibility.
    
    Returns:
    client_function (function): Function to create a FlowerClient object for a specific client.
    """
    
    def client_function(client_id):
        """
        Function to create a FlowerClient object for a specific client.
        
        Arguments:
        client_id (int): ID of the client.
        
        Returns:
        client (FlowerClient): FlowerClient object for the specified client.
        """
        
        return FlowerClient(train_dataloaders[int(client_id)], val_dataloaders[int(client_id)], 
                            input_channels, num_classes, random_seed)
    
    return client_function