import torch
from torch.utils.data import DataLoader, random_split


def prepare_dataset(dataset, batch_size, num_partitions, random_seed=42, train_ratio=0.8, val_ratio=0.2):
        """
        Creates the training, validation, and testing dataloaders for federated training, validation, and testing.
        
        Arguments:
        dataset (Dataset): Dataset object.
        batch_size (int): Batch size for dataloader.
        num_partitions (int): Number of partitions to be made to the dataset.
        random_seed (int): Random seed for reproducibility.
        train_ratio (float): Ratio of dataset to be used for training.
        
        Returns:
        train_dataloaders (List): Dataloaders for training.
        val_dataloaders (List): Dataloaders for validation.
        test_dataloader (DataLoader): Dataloader for testing.
        """
        
        generator = torch.Generator().manual_seed(random_seed)
        train_dataset, test_dataset = random_split(dataset, [train_ratio, 1 - train_ratio], generator=generator)
        
        # Splitting train_dataset and val_dataset into partitions
        num_images = len(train_dataset) // num_partitions
        partition_len = [num_images] * num_partitions
        train_datasets = random_split(train_dataset, partition_len, generator=generator)
        
        train_dataloaders = list()
        val_dataloaders = list()
        
        # Creating the train and val dataloaders
        for train_dataset in train_datasets:
            num_val = int(len(train_dataset) * val_ratio)
            num_train = len(train_dataset) - num_val
            train, val = random_split(train_dataset, [num_train, num_val], generator=generator)
            train_dataloaders.append(DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2))
            val_dataloaders.append(DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2))
        
        # Creating the test dataloader
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_dataloaders, val_dataloaders, test_dataloader