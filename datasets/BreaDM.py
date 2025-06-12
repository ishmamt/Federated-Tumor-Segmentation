import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class BreaDMDataset(Dataset):
    def __init__(self, root_dir, image_size=512):
        """
        Creates the BUSI dataset.
        
        Arguments:
        root_dir (str): Root directory of the dataset images folder.
        image_size (int): Size of the image after transformation.
        """
        
        self.image_paths = glob.glob(os.path.join(root_dir,'images','*.png'))
        self.mask_paths = glob.glob(os.path.join(root_dir,'labels','*.png'))
        self.image_size = image_size

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

        cur_mask = cv2.imread(mask_path)
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
        
        Arguments:
        idx (int): Index of the sample to get.
        
        Returns:
        image (torch.Tensor): Image tensor.
        mask (torch.Tensor): Mask tensor.
        """
        
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = self.load_image(image_path)
        mask = self.load_mask(mask_path)
        
        return torch.Tensor(image), torch.Tensor(mask)
    
if __name__ == '__main__':
    dataset = BreaDMDataset(
        root_dir='resized_data/breadm'
    )

    print(dataset.__len__())

    image, mask = dataset.__getitem__(0)

    # show_image_mask_tensor_pair(image,mask)

    # print(type(image), type(mask))