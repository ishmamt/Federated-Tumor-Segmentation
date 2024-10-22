import flwr as fl
import torch

from models.UNet import UNet


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, train_loader, val_loader, input_channels, num_classes, random_seed=42):
        """
        Creates a FlowerClient object to simulate a client.
        
        Arguments:
        train_loader (DataLoader): DataLoader for training data on a single client.
        val_loader (DataLoader): DataLoader for validation data on a single client.
        num_classes (int): Number of classes in the dataset.
        random_seed (int): Random seed for reproducibility.
        """
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        torch.manual_seed(random_seed)
        self.model = UNet(in_channels=input_channels, num_classes=num_classes)