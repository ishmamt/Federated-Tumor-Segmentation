import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BUSIDataset(Dataset):
    def __init__(self, root_dir, image_size=512):
        """
        Creates the BUSI dataset.
        
        Arguments:
        root_dir (str): Root directory of the dataset images folder.
        image_size (int): Size of the image after transfomration.
        """
        
        # Collecting all image paths and mask paths from the dataset folder
        all_images = glob.glob(os.path.join(root_dir, "**/*.png"), recursive=True)
        
        self.image_paths = list()
        self.mask_paths = list()
        self.image_size = image_size
        
        for path in all_images:
            if "normal" in path or "mask" in path:
                continue  # We are ignoring images that are "Normal" i.e. images with no tumors
            
            self.image_paths.append(path)
            current_image_mask_paths = list()  # A single image can have multiple masks for multiple tumors
            mask_path_1 = path.replace('.png', '_mask.png')
            mask_path_2 = path.replace('.png', '_mask_1.png')
            current_image_mask_paths.append(mask_path_1)
            
            if os.path.exists(mask_path_2):
                current_image_mask_paths.append(mask_path_2)
            
            self.mask_paths.append(current_image_mask_paths)
            
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()])
        
    def load_image(self, image_path):
        """
        Loads an image into a numpy array.
        
        Arguments:
        image_path (str): Path to the image file.
        
        Returns:
        image (numpy.ndarray): Numpy array representing the image.
        """
        
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image[:, :, 0]  # Convert to grayscale
        image = image / 255.0
        image = np.expand_dims(image, 0)  # Expand dimensions to include batch size 1
        image = image.astype(np.double)
        
        return image
    
    def load_mask(self, mask_path):
        """
        Loads a mask into a numpy array.
        
        Arguments:
        mask_path (str): Path to the mask file.
        
        Returns:
        mask (numpy.ndarray): Numpy array representing the mask.
        """
        
        mask = np.zeros((self.image_size, self.image_size), dtype=int)  # To combine multiple masks

        for path in mask_path:
            cur_mask = cv2.imread(path)
            cur_mask = np.mean(cur_mask, axis=-1)  # No effect for grayscale images
            cur_mask = cv2.resize(cur_mask, (self.image_size, self.image_size))
            mask[cur_mask != 0] = 1  # Combining multiple masks
            
        mask = np.expand_dims(mask, 0)  # Expand dimensions to include batch size 1
        
        return mask
    
    def __len__(self):
        """
        Pytorch Dataset method to get the number of samples in the dataset.
        
        Returns:
        length (int): Number of samples in the dataset.
        """
        
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Pytorch Dataset method to get a sample at a given index.
        
        Returns:
        image (torch.Tensor): Image tensor.
        mask (torch.Tensor): Mask tensor.
        """
        
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = self.load_image(image_path)
        mask = self.load_mask(mask_path)
        
        return torch.Tensor(image), torch.Tensor(mask)