import torch
import numpy as np
from collections import OrderedDict

from models.UNet import UNet
from train import test


def get_on_fit_config_function(cfg):
    """
    Provides an on fit config function which the server can evoke when trying to fit on the client.
    
    Arguments:
    cfg (dict): Configuration dictionary.
    
    Returns:
    on_fit_config_function (function): Function to return the on fit config for on client training.
    """
    
    def on_fit_config_function(server_round):
        """
        Provides the config for client training. The server_round is provided for customized on fit behavior.
        
        Arguments:
        server_round (int): Current server round.
        
        Returns:
        config (dict): Dictionary containing the configuration for client training.
        """
        
        return {"lr": cfg.lr, 
                "min_lr": cfg.min_lr, 
                "weight_decay": cfg.weight_decay, 
                "local_epochs": cfg.local_epochs
                }
    
    return on_fit_config_function


def get_eval_function(input_channels, num_classes, test_dataloaders, random_seed=42):
    """
    Provides an eval function which the server can evoke when trying to evaluate on the client.
    
    Arguments:
    input_channels (int): Number of input channels in the model.
    num_classes (int): Number of classes in the dataset.
    test_dataloaders (List): DataLoaders for testing on a single client.
    
    Returns:
    eval_function (function): Function to run evaluation on a client.
    """
    
    def eval_function(server_round, params, cfg):
        """
        Runs evaluation on a client. The server_round is provided for customized evaluation behavior.
        
        Arguments:
        server_round (int): Current server round.
        params (list): Parameters received from the server.
        cfg (dict): Configuration dictionary.
        
        Returns:
        loss (float): Average loss over the validation set.
        eval_metrics (dict): Additional evaluation metrics (IoU, Dice) to be sent to the server.
        """
        
        loss_list = list()
        iou_list = list()
        dice_list = list()
        
        model = UNet(in_channels=input_channels, num_classes=num_classes, random_seed=random_seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        params_dict = zip(model.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        for test_dataloader in test_dataloaders:
            loss, iou, dice = test(model, test_dataloader, device)
            
            loss_list.append(float(loss))
            iou_list.append(iou)
            dice_list.append(dice)
        
        return np.mean(loss_list), {"iou": iou_list, "dice": dice_list, "loss": loss_list}
    
    return eval_function