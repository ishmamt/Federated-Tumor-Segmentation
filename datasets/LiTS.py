import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from multiprocessing import Pool

# from utils import show_image_mask_tensor_pair

class LiTSDataset(Dataset):
    def __init__(self, root_dir, image_size=512):
        """
        Creates the LiTS dataset.

        Arguments:
        root_dir (str): Root directory of the dataset images folder.
        image_size (int): Size of the image after transformation.
        target_threshold (float): Threshold for the percentage of target label (tumor) present in the image.
        """
        
        self.image_size = image_size

        self.image_mask_list = self.get_image_mask_pairs(root_dir)
        
        # Use a Pool to parallelize the process of keeping images with more than 5% non-zero pixels in the mask
        # with Pool() as pool:
        #     self.image_mask_list = pool.map(self.process_file, self.get_image_mask_pairs(root_dir))
            
        # self.image_mask_list = [file for file in self.image_mask_list if file is not None]
    
    def get_image_mask_pairs(self, root_dir):
        """
        Fetches all image mask pairs from the specified directory.

        Arguments:
        root_dir (str): Root directory of the dataset images folder.

        Returns:
        image_mask_list (list): List of image mask pair file paths.
        """
        
        images_dir = os.path.join(root_dir, "train_images")
        masks_dir = os.path.join(root_dir, "train_masks")
        
        image_mask_list = [{"image": os.path.join(images_dir, image_name), "mask": os.path.join(masks_dir, image_name)}
        for image_name in os.listdir(images_dir) if image_name.endswith(".jpg") and os.path.exists(os.path.join(masks_dir, image_name))]
        
        return image_mask_list
    
    def load_image(self, image_path):
        """
        Transforms the image into a numpy array.
        
        Arguments:
        image_path (str): Path to the image file.
        
        Returns:
        image (numpy.ndarray): Numpy array representing the transformed image.
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
        Transforms the mask into a numpy array.
        
        Arguments:
        mask_path (str): Path to the mask file.
        
        Returns:
        mask (numpy.ndarray): Numpy array representing the mask.
        """

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask == 2.0, 255.0, 0.0)  # The tumor is presented as 255 and the background as 0
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask = mask / 255.0
        mask = np.expand_dims(mask, 0)  # Expand dimensions to include batch size 1
        mask = mask.astype(np.double)
        
        return mask
        
    def __len__(self):
        """
        Pytorch Dataset method to get the number of samples in the dataset.
        
        Returns:
        length (int): Number of samples in the dataset.
        """
        
        return len(self.image_mask_list)
    
    def __getitem__(self, idx):
        """
        Pytorch Dataset method to get a sample at a given index.
        
        Arguments:
        idx (int): Index of the sample to get.
        
        Returns:
        image (torch.Tensor): Image tensor.
        mask (torch.Tensor): Mask tensor.
        """
        
        # self.image_mask_list

        image = self.load_image(self.image_mask_list[idx]["image"])
        mask = self.load_mask(self.image_mask_list[idx]["mask"])
        
        return torch.Tensor(image), torch.Tensor(mask)
    

if __name__ == '__main__':
    dataset = LiTSDataset(
        root_dir='resized_data/lits'
    )

    print(dataset.__len__())

    image, mask = dataset.__getitem__(100)
    # show_image_mask_tensor_pair(image,mask)
    # print(type(image),type(mask))