import os
import torch
import numpy as np
from collections import OrderedDict

from models.unetWithModifications import SharedUnet,PersonalizedUnet,\
SharedDown,QGenerator,SharedUpWithAttn
from trainWithModifications import testWithAttention,testWithAdapter


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

    def setParamsWithAttention(self, modelDown, modelUp, params):
      """
      Updates the model parameters using the given numpy arrays.
      
      Arguments:
      params (list): List of numpy arrays representing the model parameters.
      """
      
      num_params_down = len(list(modelDown.parameters()))
      params_down = params[:num_params_down]  # First half for Model A
      params_up = params[num_params_down:]  # Second half for Model B

      # Load into Model A
      for param, model_param in zip(params_down, modelDown.parameters()):
          model_param.data = torch.tensor(param)

      # Load into Model B
      for param, model_param in zip(params_up, modelUp.parameters()):
          model_param.data = torch.tensor(param)

      return (modelDown,modelUp)

    def evalFunctuionWithAttention(server_round, params, cfg):
      loss_list = list()
      iou_list = list()
      dice_list = list()
      
      modelDown = SharedDown(
        in_channels=input_channels, 
        random_seed=random_seed
      )
      outputChannelsOfModelDown = modelDown.output_channels
      modelUp = SharedUpWithAttn(
        in_channels=outputChannelsOfModelDown,
        num_classes=num_classes,
        random_seed=random_seed
      )
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
      num_params_down = len(list(modelDown.parameters()))
      params_down = params[:num_params_down]  # First half for Model A
      params_up = params[num_params_down:]  # Second half for Model B

      # Load into Model A
      for param, model_param in zip(params_down, modelDown.parameters()):
          model_param.data = torch.tensor(param)

      # Load into Model B
      for param, model_param in zip(params_up, modelUp.parameters()):
          model_param.data = torch.tensor(param)

      numberOfqGenerators = len(test_dataloaders)
      qGenerators = []
      for idx in range(numberOfqGenerators):
        q_weight_path = \
        f'/content/drive/MyDrive/UFF/Federated-Tumor-Segmentation/q_weight/model{idx}.pth'
        qGenerators.append(QGenerator(
          in_channels = outputChannelsOfModelDown,
          random_seed = random_seed
        ))

        if os.path.exists(q_weight_path):
            print(f"Loading saved qGenerator {idx}'s weights...")
            qGenerators[-1].load_state_dict(torch.load(q_weight_path))
        else:
            print(f"No saved weights found, skipping qGenerator {idx}'s weight loading.")

      for idx,test_dataloader in enumerate(test_dataloaders):
          loss, iou, dice = testWithAttention(
            (modelDown,qGenerators[idx],modelUp),
            test_dataloader, 
            device
          )
          
          loss_list.append(float(loss))
          iou_list.append(iou)
          dice_list.append(dice)
      
      return np.mean(loss_list), {"iou": iou_list, "dice": dice_list, "loss": loss_list}
    
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
        
        model = SharedUnet(in_channels=input_channels, num_classes=num_classes, random_seed=random_seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        params_dict = zip(model.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        numberOfAdapters = len(test_dataloaders)
        adapters = []
        for idx in range(numberOfAdapters):
          adapter_weight_path = \
          f'/content/drive/MyDrive/UFF/Federated-Tumor-Segmentation/adapter_weight/model{idx}.pth'
          adapters.append(PersonalizedUnet(
            in_channels = model.featureSizeForAdapter,
            num_classes = num_classes,
            random_seed = random_seed
          ))

          if os.path.exists(adapter_weight_path):
              print("Loading saved adapter weights...")
              adapters[-1].load_state_dict(torch.load(adapter_weight_path))
          else:
              print("No saved weights found, skipping adapter weight loading.")

        for idx,test_dataloader in enumerate(test_dataloaders):
            loss, iou, dice = testWithAdapter(model, adapters[idx], test_dataloader, device)
            
            loss_list.append(float(loss))
            iou_list.append(iou)
            dice_list.append(dice)
        
        return np.mean(loss_list), {"iou": iou_list, "dice": dice_list, "loss": loss_list}
    
    return evalFunctuionWithAttention