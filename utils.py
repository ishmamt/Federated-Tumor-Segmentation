import torch
import os
import sys
import numpy as np
from scipy import stats
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
import json

#The followingb function is required for calculating confidence scores of our chosen methods
def calculate_confidence_scores(method_name):

  conf_json_path = 'outputs/conf.json'
  
  if os.path.exists(conf_json_path):
    with open(conf_json_path, 'r') as file:
      data = json.load(file)
  else:
    data = {}
  
  data[method_name] = {}

  results = []
  num_clients = 3
  num_runs = 5
  dice_score_dict = {}
  
  for idx in range(num_clients):
    dice_score_dict[idx] = []
    for ix in range(num_runs):
      dice_json_path = os.path.join('outputs',method_name,f'results{ix+1}.json')
      with open(dice_json_path, 'r') as file:
        dice_score = json.load(file)
        
        dice_score_dict[idx].append(dice_score[str(idx)])

  for ix in range(num_clients):
    mean = np.mean(dice_score_dict[ix])
    sem = stats.sem(dice_score_dict[ix])
    confidence = 0.95
    n = len(dice_score_dict[ix])
    df = n - 1
    t_critical = stats.t.ppf((1 + confidence) / 2, df)
    margin_of_error = t_critical * sem
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    data[method_name][str(ix)] = f'The {method_name} has dice score {round(mean,4)} +- {round(upper_bound-mean,4)} with 95\% conf score'
  
  with open('outputs/conf.json', 'w') as file:
    json.dump(data, file)

  return data


def convert_json_dict_to_txt(json_file_path):
    with open(json_file_path, 'r') as file:
      data = json.load(file)

    with open('outputs/conf_score.txt','w') as f:
        for outer_key in data:
            for inner_key in data[outer_key]:
                f.write(f'{data[outer_key][inner_key]} for client {inner_key}\n')
            


#The following function is required for IGC in FedDP
@torch.no_grad()
def compute_pred_uncertainty(net_clients, images):
    preds = []
    for net in net_clients:
        pred = net(images)
        b, c, h, w = pred.size()
        preds.append(pred.unsqueeze(0))
    preds = torch.cat(preds, dim=0)

    umap = torch.std(preds, dim=0)
    # print(umap.max())
    if umap.max() > 0:
        umap = umap / umap.max()
    # umap = umap.view(b, c, h * w)
    # umap = torch.softmax(umap, dim=-1)
    umap = umap.view(b, c, h, w)
    return umap, preds

#The following function is reqired for IGC in FedDP
def weighted_bce_loss(score, target, weight):
    target = target.float()
    smooth = 1e-5
    score = torch.clamp(score, smooth, 1 - smooth)

    loss = -(target * torch.log(score) +
             (1 - target) * torch.log(1 - score)) * weight
    loss = torch.mean(loss)
    return loss

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
    

class Logger():
    def __init__(self, save_dir):
        """
        A custom logger for tracking training and validation metrics.
        
        Arguments:
        save_dir (str): Directory to save the log file.
        """
        
        self.save_dir = save_dir
        self.log = OrderedDict([
            ("epoch", []),
            ("lr", []),
            ("train_loss", []),
            ("train_iou", []),
            ("train_dice", []),
            ("val_loss", []),
            ("val_iou", []),
            ("val_dice", [])
            ])
    
    def save(self):
        """
        Saves the log data to a CSV file in the specified directory.
        """
        
        pd.DataFrame(self.log).to_csv(os.path.join(self.save_dir, "log.csv"), index=False)
        
    def update(self, epoch, lr, train_loss, train_iou, train_dice, val_loss, val_iou, val_dice):
        """
        Updates the log data with the current epoch, learning rate, training loss, training IoU, training Dice, validation loss, validation IoU, and validation Dice.
        
        Arguments:
        epoch (int): Current epoch number.
        lr (float): Current learning rate.
        train_loss (float): Training loss.
        train_iou (float): Training IoU.
        train_dice (float): Training Dice score.
        val_loss (float): Validation loss.
        val_iou (float): Validation IoU.
        val_dice (float): Validation Dice score.
        """
        
        self.log["epoch"].append(epoch)
        self.log["lr"].append(lr)
        self.log["train_loss"].append(train_loss)
        self.log["train_iou"].append(train_iou)
        self.log["train_dice"].append(train_dice)
        self.log["val_loss"].append(val_loss)
        self.log["val_iou"].append(val_iou)
        self.log["val_dice"].append(val_dice)
        
        self.save()
    
        
class AvgMeter():
    def __init__(self):
        """
        A custom meter for calculating averages.
        """
        
        self.reset()

    def reset(self):
        """
        Resets the meter to its initial state.
        """
        
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        """
        Updates the meter with the current value.
        
        Arguments:
        val (float): Current value.
        n (int): Number of occurrences.
        """
        
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
  # method_names = ['fedAVG', 'fedAVGM', 'fedADAGRAD', 'fedPER', 'fedREP','fedDP', 'fedOAP']
  # for name in method_names:
  #   calculate_confidence_scores(name)
  convert_json_dict_to_txt('outputs/conf.json')