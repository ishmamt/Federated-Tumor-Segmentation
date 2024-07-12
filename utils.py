import os
import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt


def show_image(img, msk, labels=['mask'], semantic=False, threshold=False):
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
        pd.DataFrame(self.log).to_csv(os.path.join(self.save_dir, "log.csv"), index=False)
        
    def update(self, epoch, lr, train_loss, train_iou, train_dice, val_loss, val_iou, val_dice):
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
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count