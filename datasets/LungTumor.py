import pickle
import numpy as np
from PIL import Image
import os
import pandas as pd
from PIL import Image
# import torchvision.transforms as transforms

import glob
import cv2
import torch
from torch.utils.data import Dataset

from utils import show_image_mask_tensor_pair

def to_uint8(img):
    img = np.array(img, dtype=np.float32)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    return img.astype(np.uint8)

def to_rgb(img):
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    return img

def convert_pkl_pngs(pkl_paths):

    data0 = pd.read_pickle(pkl_paths[0])
    data1 = pd.read_pickle(pkl_paths[1])
    data = pd.concat([data0, data1], ignore_index=True)

    for idx, row in data.iterrows():
        # Convert to uint8 format
        image = to_uint8(row['hu_array'])      # CT image (thresholded)
        mask = to_uint8(row['mask'])           # Binary mask (0 or 1 → 0 or 255)

        image_rgb = to_rgb(image)
        mask_rgb = to_rgb(mask)

        # Save images
        Image.fromarray(image_rgb).save(f'datasets/lung_tumor_segmentation/images/{idx:04d}.png')
        Image.fromarray(mask_rgb).save(f'datasets/lung_tumor_segmentation/labels/{idx:04d}.png')

class LungTumorDataset(Dataset):
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
    
    # convert_pkl_pngs(['datasets/lung_tumor_segmentation/lung_cancer_train.pkl','datasets/lung_tumor_segmentation/lung_cancer_test.pkl'])
    dd = LungTumorDataset(root_dir='datasets/lung_tumor_segmentation')
    print(dd.__len__())
    tensor1,tensor2 = dd.__getitem__(400)
    show_image_mask_tensor_pair(tensor1,tensor2)
