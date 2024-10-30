import os
import numpy as np
import cv2
import h5py
import torch
from torch.utils.data import Dataset
from multiprocessing import Pool


class BRATSDataset(Dataset):
    def __init__(self, root_dir, image_size=512):
        """
        Creates the BRATS dataset.

        Arguments:
        root_dir (str): Root directory of the dataset images folder.
        image_size (int): Size of the image after transfomration.
        """
        
        self.image_size = image_size
        
        # Use a Pool to parallelize the process of keeping images with more than 5% non-zero pixels in the mask
        with Pool() as pool:
            self.file_paths = pool.map(self.process_file, self.get_h5_files(root_dir))
            
        self.file_paths = [file for file in self.file_paths if file is not None]
    
    def get_h5_files(self, root_dir):
        """
        Fetches all.h5 files from the specified directory.

        Arguments:
        root_dir (str): Root directory of the dataset images folder.

        Returns:
        h5_files (list): List of .h5 file paths.
        """
        
        return [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".h5")]
    
    def process_file(self, h5_file):
        """
        Only considers images with more than 5% non-zero pixels in the mask.
        
        Arguments:
        h5_file (str): Path to the.h5 file.
        
        Returns:
        h5_file (str) or None: Path to the.h5 file or None if the image does not meet the criteria.
        """
        
        with h5py.File(h5_file, "r") as file:
            mask = self.load_mask(file["mask"][()])
            
            if (((mask != 0).sum() / (self.image_size * self.image_size)) * 100) >= 5.0:
                return h5_file
            else:
                return None
    
    def load_image(self, image):
        """
        Transforms the image into a numpy array.
        
        Arguments:
        image (numpy.ndarray): Numpy array representing the image.
        
        Returns:
        image (numpy.ndarray): Numpy array representing the transformed image.
        """
        
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image[:, :, 0]  # Convert to grayscale
        image = image / 255.0
        image = np.expand_dims(image, 0)  # Expand dimensions to include batch size 1
        image = image.astype(np.double)
        
        return image
    
    def load_mask(self, mask):
        """
        Transforms the mask into a numpy array.
        
        Arguments:
        mask (numpy.ndarray): Numpy array representing the mask.
        
        Returns:
        mask (numpy.ndarray): Numpy array representing the mask.
        """

        mask = mask.transpose((2, 0, 1))
        new_mask = np.zeros((self.image_size, self.image_size), dtype=int)  # To combine multiple masks

        for cur_mask in [mask[0], mask[1], mask[2]]:
            cur_mask = cv2.resize(cur_mask, (self.image_size, self.image_size))
            new_mask[cur_mask != 0] = 1  # Combining multiple masks
            
        new_mask = np.expand_dims(new_mask, 0)  # Expand dimensions to include batch size 1
        
        return new_mask
        
    def __len__(self):
        """
        Pytorch Dataset method to get the number of samples in the dataset.
        
        Returns:
        length (int): Number of samples in the dataset.
        """
        
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """
        Pytorch Dataset method to get a sample at a given index.
        
        Arguments:
        idx (int): Index of the sample to get.

        Returns:
        image (torch.Tensor): Image tensor.
        mask (torch.Tensor): Mask tensor.
        """

        with h5py.File(self.file_paths[idx], "r") as file:
            image = self.load_image(file["image"][()])
            mask = self.load_mask(file["mask"][()])
            
        return torch.Tensor(image), torch.Tensor(mask)