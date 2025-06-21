import os
import torch
import numpy as np
from collections import OrderedDict

from flower.client import FlowerClientFedOAP, FlowerClientFedDP, FlowerClientFedREP, FlowerClientFedAVG

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
        
        return {"lr": cfg['lr'], 
                "min_lr": cfg['min_lr'], 
                "weight_decay": cfg['weight_decay'], 
                "local_epochs": cfg['local_epochs'],
                "rep_epochs": cfg['rep_epochs']
                }
    
    return on_fit_config_function


def get_eval_function(strategy, input_channels, num_classes, val_dataloaders, output_dir, random_seed=42):
    """
    Provides an eval function which the server can evoke when trying to evaluate on the client.
    
    Arguments:
    input_channels (int): Number of input channels in the model.
    num_classes (int): Number of classes in the dataset.
    test_dataloaders (List): DataLoaders for testing on a single client.
    
    Returns:
    eval_function (function): Function to run evaluation on a client.
    """

    def evalFunctionFedOAP(server_round, params, cfg):

      loss_list = list()
      iou_list = list()
      dice_list = list()

      for idx,val_dataloader in enumerate(val_dataloaders):

        modelServerEval = FlowerClientFedOAP(
          client_id=idx,
          train_dataloader=None,
          val_dataloader=val_dataloader,
          input_channels=input_channels,
          num_classes=num_classes,
          output_dir=output_dir
        )

        loss, lenVal, accDict = modelServerEval.evaluate(params,{})

        loss_list.append(float(loss))
        iou_list.append(accDict['iou'])
        dice_list.append(accDict['dice'])

      return np.mean(loss_list), {"iou": iou_list, "dice": dice_list, "loss": loss_list}
    

    def evalFunctuionFedDP(server_round, params, cfg):
      loss_list = list()
      iou_list = list()
      dice_list = list()

      for idx,val_dataloader in enumerate(val_dataloaders):

        modelServerEval = FlowerClientFedDP(
          client_id=idx,
          train_dataloader=None,
          val_dataloader=val_dataloader,
          input_channels=input_channels,
          num_classes=num_classes,
          output_dir=output_dir
        )

        loss, lenVal, accDict = modelServerEval.evaluate(params,{})

        loss_list.append(float(loss))
        iou_list.append(accDict['iou'])
        dice_list.append(accDict['dice'])

      return np.mean(loss_list), {"iou": iou_list, "dice": dice_list, "loss": loss_list}
    

    def evalFunctionFedREP(server_round, params, cfg):
      loss_list = list()
      iou_list = list()
      dice_list = list()

      for idx,val_dataloader in enumerate(val_dataloaders):

        modelServerEval = FlowerClientFedREP(
          client_id = idx,  
          train_dataloader=None, 
          val_dataloader=val_dataloader, 
          input_channels=input_channels, 
          num_classes=num_classes, 
          output_dir=output_dir, 
          random_seed=random_seed
        )

        loss, lenVal, accDict = modelServerEval.evaluate(params,{})

        loss_list.append(float(loss))
        iou_list.append(accDict['iou'])
        dice_list.append(accDict['dice'])

      return np.mean(loss_list), {"iou": iou_list, "dice": dice_list, "loss": loss_list}


    def evalFunctionFedAVG(server_round, params, cfg):
      loss_list = list()
      iou_list = list()
      dice_list = list()

      for idx,val_dataloader in enumerate(val_dataloaders):

        modelServerEval = FlowerClientFedAVG(
          client_id = idx,  
          train_dataloader=None, 
          val_dataloader=val_dataloader, 
          input_channels=input_channels, 
          num_classes=num_classes, 
          output_dir=output_dir, 
          random_seed=random_seed
        )

        loss, lenVal, accDict = modelServerEval.evaluate(params,{})

        loss_list.append(float(loss))
        iou_list.append(accDict['iou'])
        dice_list.append(accDict['dice'])

      return np.mean(loss_list), {"iou": iou_list, "dice": dice_list, "loss": loss_list}
    

    if strategy == 'fedOAP':
      return evalFunctionFedOAP
    elif strategy == 'fedDP':
      return evalFunctuionFedDP
    elif strategy == 'fedREP':
      return evalFunctionFedREP
    elif strategy == 'fedAVG':
      return evalFunctionFedAVG
    elif strategy == 'fedADAGRAD':
      return evalFunctionFedAVG
    else:
      print('The given strategy is yet to be implemented')
      exit()