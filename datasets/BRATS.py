import os
import numpy as np
import cv2
import h5py
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class BRATSDataset(Dataset):
    def __init__(self, root_dir, image_size=512):
        """
        Creates the BRATS dataset.

        Arguments:
        root_dir (str): Root directory of the dataset images folder.
        image_size (int): Size of the image after transfomration.
        """
        
        self.file_paths = self.get_h5_files(root_dir)
        self.image_size = image_size
    
    def get_h5_files(self, root_dir):
        """
        Fetches all.h5 files from the specified directory.

        Arguments:
        root_dir (str): Root directory of the dataset images folder.

        Returns:
        h5_files (list): List of .h5 file paths.
        """
        
        return [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.h5')]
    
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

        with h5py.File(self.file_paths[idx], 'r') as file:
            image = self.load_image(file['image'][()])
            mask = self.load_mask(file['mask'][()])
            
        return torch.Tensor(image), torch.Tensor(mask)
    
def show_image(img, msk, labels=['mask'], semantic=False, threshold=False):
    """
    Displays an image and its corresponding mask.
    
    Arguments:
    img (np.ndarray): Input image.
    msk (np.ndarray): Input mask.
    labels (list): Labels for the mask channels.
    semantic (bool): Whether the mask represents semantic segmentation.
    threshold (bool): Whether the mask represents thresholding.
    """
    
    fig, axs = plt.subplots(1, 1 + len(labels), figsize=(8, 3))
    img = np.squeeze(img)

    axs[0].imshow(img, cmap='gray')
    axs[0].set_title("image")

    for i in range(len(labels)):
        if semantic and threshold:
            # Create a binary mask based on the channel with the highest value
            binary_mask = np.argmax(msk, axis=0) == i
            axs[i + 1].imshow(binary_mask, vmin=0, vmax=1, cmap='gray')
        else:
            axs[i + 1].imshow(msk[i], vmin=0, vmax=1, cmap='gray')
        axs[i + 1].set_title(f"{labels[i]}")

    plt.show()

if __name__ == "__main__":
    # directory = os.path.join('..', 'BraTS2020_training_data', 'content', 'data')
    directory = "G:/federatedLearningDatasets/BraTS2020_training_data/content/data"
    train_dataset = BRATSDataset(directory, 128)

    for i in range(25):
        image, mask = train_dataset[i]
        show_image(image, mask)