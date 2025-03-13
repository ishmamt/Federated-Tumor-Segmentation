import torch
from torch.utils.data import DataLoader, random_split

from datasets.BUSI import BUSIDataset
from datasets.BRATS import BRATSDataset
from datasets.LiTS import LiTSDataset


def prepare_datasets(datasets, batch_size, num_clients, random_seed=42, train_ratio=0.8, val_ratio=0.2):
        """
        Creates the training, validation, and testing dataloaders for federated training, validation, and testing.
        
        Arguments:
        datasets (List): List of Dataset objects.
        batch_size (int): Batch size for dataloaders.
        num_clients (int): Number of clients.
        random_seed (int): Random seed for reproducibility.
        train_ratio (float): Ratio of dataset to be used for training.
        val_ratio (float): Ratio of dataset to be used for validation.
        
        Returns:
        train_dataloaders (List): Dataloaders for training.
        val_dataloaders (List): Dataloaders for validation.
        test_dataloaders (List): Dataloaders for testing.
        """
        
        generator = torch.Generator().manual_seed(random_seed)  # Setting the random seed for reproducibility
        train_dataloaders = list()
        val_dataloaders = list()
        test_dataloaders = list()
        
        for dataset in datasets:
            train_dataset, test_dataset = random_split(dataset, [train_ratio, 1 - train_ratio], generator=generator)
            
            # Creating the train and val dataloaders
            num_val = int(len(train_dataset) * val_ratio)
            num_train = len(train_dataset) - num_val
            train, val = random_split(train_dataset, [num_train, num_val], generator=generator)
            train_dataloaders.append(DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2))
            val_dataloaders.append(DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2))
            
            # Creating the test dataloader
            test_dataloaders.append(DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2))
        
        return train_dataloaders, val_dataloaders, test_dataloaders
    

def load_datasets(dataset_dirs, image_size):
    """
    Loads the datasets from the specified list of directories.
    
    Arguments:
    dataset_dirs (list): List of directories containing the datasets.
    image_size (int): Size of the image after transformation.
    
    Returns:
    datasets (list): List of Dataset objects.
    """
    
    datasets = list()
    for dataset_name in dataset_dirs:
        if dataset_name == "busi":
            datasets.append(BUSIDataset(dataset_dirs[dataset_name], image_size))
        
        elif dataset_name == "brats":
            datasets.append(BRATSDataset(dataset_dirs[dataset_name], image_size))
            
        elif dataset_name == "lits":
            datasets.append(LiTSDataset(dataset_dirs[dataset_name], image_size))
    
    return datasets
