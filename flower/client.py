import json
import os
import shutil
import glob
import traceback
import flwr as fl
import torch
from collections import OrderedDict
from torch.optim import Adam,AdamW,lr_scheduler

from models.UNet import UNet
from models.fedREP import UnetFedRep
from models.fedPER import UnetFedPer
from models.fedDP import UNetWithAttention
from models.fedOAP import UNetWithCrossAttention
from train import train, test, fedrep_train, fedrep_test

class FlowerClientFedOAP(fl.client.NumPyClient):
  def __init__(self, client_id, train_dataloader, val_dataloader, input_channels, num_classes, output_dir, random_seed=42):
      """
      Creates a FlowerClient with self attention module to simulate a client.
      
      Arguments:
      train_dataloader (DataLoader): DataLoader for training data on a single client.
      val_dataloader (DataLoader): DataLoader for validation data on a single client.
      input_channels (int): Number of input channels in the model.
      num_classes (int): Number of classes in the dataset.
      random_seed (int): Random seed for reproducibility.
      """
      
      super().__init__()
      self.client_id = client_id
      self.train_dataloader = train_dataloader
      self.val_dataloader = val_dataloader
      self.num_classes = num_classes
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
      self.model = UNetWithCrossAttention(
          in_channels = input_channels,
          num_classes = num_classes
      )

      self.output_dir = output_dir
      self.temporaryWeightsPath = 'temporaryWeights'
      self.bottleneckQWeightsPath = os.path.join(self.temporaryWeightsPath,f'fedOAPbottleneckQ{client_id}.pth')
      self.queryWeightsPath = os.path.join(self.temporaryWeightsPath,f'fedOAPquery{client_id}.pth')
      self.adapterWeightsPath = os.path.join(self.temporaryWeightsPath,f'fedOAPadapter{client_id}.pth')
      self.outWeightsPath = os.path.join(self.temporaryWeightsPath,f'fedOAPout{client_id}.pth')
      self.modelWeightsPath = os.path.join(self.temporaryWeightsPath,'fedOAPserver.pth')
  
  def setPersonalizedParameters(self):
      
      if os.path.exists(self.queryWeightsPath) and os.path.exists(self.bottleneckQWeightsPath):
          print('Loading personalized parameters')
          try:
            self.model.bottleneckQ.load_state_dict(torch.load(self.bottleneckQWeightsPath))
            self.model.cross_attn.query.load_state_dict(torch.load(self.queryWeightsPath))
            self.model.adapter.load_state_dict(torch.load(self.adapterWeightsPath))
            self.model.outc.load_state_dict(torch.load(self.outWeightsPath))
          except Exception as e:
            print('Problem while loading personalized parameters')
            traceback.print_exc()
            exit()
      else:
          print('bottleneckQ and Query parameters are not saved')

  def set_parameters(self, params):
      """
      Updates the model parameters using the given numpy arrays.
      
      Arguments:
      params (list): List of numpy arrays representing the model parameters.
      """
      
      params_dict = zip(self.model.state_dict().keys(), params)
      state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
      self.model.load_state_dict(state_dict, strict=True)
  
  def saveServerModel(self):
      print('saving server (fedOAP) weights')
      try:
        torch.save(self.model.state_dict(),self.modelWeightsPath)
      except Exception as e:
        print('Exception while saving server weights')
        traceback.print_exc()
        exit()

  def getPersonalizedParameters(self):
      print('saving personalized weights')
      try:
        torch.save(self.model.bottleneckQ.state_dict(), self.bottleneckQWeightsPath)
        torch.save(self.model.cross_attn.query.state_dict(), self.queryWeightsPath)
        torch.save(self.model.adapter.state_dict(), self.adapterWeightsPath)
        torch.save(self.model.outc.state_dict(), self.outWeightsPath)
      except Exception as e:
        print('Exception while saving personalized weights')
        traceback.print_exc()
        exit()

  def get_parameters(self, config):
      """
      Returns the current model parameters as numpy arrays.
      
      Arguments:
      cfg (dict): Configuration dictionary.
      
      Returns:
      params (list): List of numpy arrays representing the model parameters.
      """
      return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
  
  def copy_best_weights(self):
    temporaryWeightsPaths = glob.glob(f'temporaryWeights/*{self.client_id}.pth')
    for path in temporaryWeightsPaths:
      file_name = path.split('/')[-1]
      shutil.copy2(path, os.path.join(self.output_dir,file_name))
    if os.path.exists('temporaryWeights/fedOAPserver.pth'):
      shutil.copy2('temporaryWeights/fedOAPserver.pth', os.path.join(self.output_dir,'fedOAPserver.pth'))

  def fit(self, params, cfg):
      """
      Trains model on a client using the client's data.
      
      Arguments:
      params (list): List of numpy arrays representing the model parameters.
      cfg (dict): Configuration dictionary.
      
      Returns:
      params (list): List of numpy arrays representing the updated model parameters.
      length (int): Length of the training dataloader.
      Dict (dict): Additional information (Training history) to be sent to the server.
      """
      
      self.set_parameters(params)  # Update model parameters from the server
      self.setPersonalizedParameters()

      optimizer = AdamW(self.model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
      scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10, 
        T_mult=2, 
        eta_min=cfg['min_lr']
      )
      
      # Local training
      history = train(
          model = self.model,
          train_dataloader = self.train_dataloader, 
          optimizer = optimizer, 
          scheduler = scheduler, 
          epochs = cfg["local_epochs"], 
          device = self.device
        )
      
      self.getPersonalizedParameters()
      self.saveServerModel()

      # Length of dataloader is for FedAVG, dict is for any additional info sent to the server
      return self.get_parameters({}), len(self.train_dataloader), {"history": history}
  
  def evaluate(self, params, cfg):
      """
      Evaluates the model on the parameters sent by the server.
      
      Arguments:
      params (list): List of numpy arrays representing the model parameters.
      cfg (dict): Configuration dictionary.
      
      Returns:
      loss (float): Average loss over the validation set.
      length (int): Length of the validation dataloader.
      eval_metrics (dict): Additional evaluation metrics (IoU, Dice) to be sent to the server.
      """
      
      self.set_parameters(params)  # Update model parameters from the server
      self.setPersonalizedParameters()
      
      loss, iou, dice = test(
        self.model, 
        self.val_dataloader, 
        self.device
      )

      best_dice_path = os.path.join(self.output_dir,'best_dice.json')
      if os.path.exists(best_dice_path):
        with open(best_dice_path, "r") as f:
          best_dice_dict = json.load(f)
      else:
        best_dice_dict = {
          '0':-1.0,
          '1':-1.0,
          '2':-1.0
        }
        with open(best_dice_path, "w") as f:
            json.dump(best_dice_dict, f, indent=4)
    
      if dice > best_dice_dict[str(self.client_id)]:
        best_dice_dict[str(self.client_id)] = dice
        self.copy_best_weights()
        with open(best_dice_path, "w") as f:
            json.dump(best_dice_dict, f, indent=4)
              
      return float(loss), len(self.val_dataloader), {"iou": iou, "dice": dice}



class FlowerClientFedDP(fl.client.NumPyClient):
  def __init__(self, client_id, train_dataloader, val_dataloader, input_channels, num_classes, output_dir, random_seed=42):
      """
      Creates a FlowerClient with self attention module to simulate a client.
      
      Arguments:
      train_dataloader (DataLoader): DataLoader for training data on a single client.
      val_dataloader (DataLoader): DataLoader for validation data on a single client.
      input_channels (int): Number of input channels in the model.
      num_classes (int): Number of classes in the dataset.
      random_seed (int): Random seed for reproducibility.
      """
      
      super().__init__()
      self.client_id = client_id
      self.train_dataloader = train_dataloader
      self.val_dataloader = val_dataloader
      self.num_classes = num_classes
      self.output_dir = output_dir
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
      self.model = UNetWithAttention(
          in_channels = input_channels,
          num_classes = num_classes
      )

      self.temporaryWeightsPath = 'temporaryWeights'
      self.queryWeightsPath = os.path.join(self.temporaryWeightsPath,f'fedDPquery{self.client_id}.pth')
      self.modelWeightsPath = os.path.join(self.temporaryWeightsPath,f'fedDPserver.pth')

  def setQueryParameters(self):
      
      if os.path.exists(self.queryWeightsPath):
          print('Loading query parameters')
          try:
            self.model.attn.query.load_state_dict(torch.load(self.queryWeightsPath))
          except Exception as e:
            print('Problem while loading query parameters')
            traceback.print_exc()
            exit()
      else:
          print('Query parameter not saved')

  def set_parameters(self, params):
      """
      Updates the model parameters using the given numpy arrays.
      
      Arguments:
      params (list): List of numpy arrays representing the model parameters.
      """
      
      params_dict = zip(self.model.state_dict().keys(), params)
      state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
      self.model.load_state_dict(state_dict, strict=True)
  
  def saveServerModel(self):
      print('saving fedDP server weights')
      try:
        torch.save(self.model.state_dict(),self.modelWeightsPath)
      except Exception as e:
        print('Exception while saving server weights')
        traceback.print_exc()
        exit()

  def getQueryParameters(self):
      print('saving query weights')
      try:
        torch.save(self.model.attn.query.state_dict(), self.queryWeightsPath)
      except Exception as e:
        print('Exception while saving query weights')
        traceback.print_exc()
        exit()

  def get_parameters(self, config):
      """
      Returns the current model parameters as numpy arrays.
      
      Arguments:
      cfg (dict): Configuration dictionary.
      
      Returns:
      params (list): List of numpy arrays representing the model parameters.
      """
      return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
  
  def copy_best_weights(self):
    temporaryWeightsPaths = glob.glob(f'temporaryWeights/*{self.client_id}.pth')
    for path in temporaryWeightsPaths:
      file_name = path.split('/')[-1]
      shutil.copy2(path, os.path.join(self.output_dir,file_name))
    if os.path.exists('temporaryWeights/fedDPserver.pth'):
      shutil.copy2('temporaryWeights/fedDPserver.pth', os.path.join(self.output_dir,'fedDPserver.pth'))

  def fit(self, params, cfg):
      """
      Trains model on a client using the client's data.
      
      Arguments:
      params (list): List of numpy arrays representing the model parameters.
      cfg (dict): Configuration dictionary.
      
      Returns:
      params (list): List of numpy arrays representing the updated model parameters.
      length (int): Length of the training dataloader.
      Dict (dict): Additional information (Training history) to be sent to the server.
      """
      
      self.set_parameters(params)  # Update model parameters from the server
      self.setQueryParameters()

      optimizer = AdamW(self.model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
      scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10, 
        T_mult=2, 
        eta_min=cfg['min_lr']
      )
      
      # Local training
      history = train(
          model = self.model,
          train_dataloader = self.train_dataloader, 
          optimizer = optimizer, 
          scheduler = scheduler, 
          epochs = cfg["local_epochs"], 
          device = self.device
        )
      
      self.getQueryParameters()
      self.saveServerModel()

      # Length of dataloader is for FedAVG, dict is for any additional info sent to the server
      return self.get_parameters({}), len(self.train_dataloader), {"history": history}
  
  def evaluate(self, params, cfg):
      """
      Evaluates the model on the parameters sent by the server.
      
      Arguments:
      params (list): List of numpy arrays representing the model parameters.
      cfg (dict): Configuration dictionary.
      
      Returns:
      loss (float): Average loss over the validation set.
      length (int): Length of the validation dataloader.
      eval_metrics (dict): Additional evaluation metrics (IoU, Dice) to be sent to the server.
      """
      
      self.set_parameters(params)  # Update model parameters from the server
      self.setQueryParameters()
      
      loss, iou, dice = test(
          self.model, 
          self.val_dataloader, 
          self.device
        )

      best_dice_path = os.path.join(self.output_dir,'best_dice.json')
      if os.path.exists(best_dice_path):
        with open(best_dice_path, "r") as f:
          best_dice_dict = json.load(f)
      else:
        best_dice_dict = {
          '0':-1.0,
          '1':-1.0,
          '2':-1.0
        }
        with open(best_dice_path, "w") as f:
            json.dump(best_dice_dict, f, indent=4)
    
      if dice > best_dice_dict[str(self.client_id)]:
        best_dice_dict[str(self.client_id)] = dice
        self.copy_best_weights()
        with open(best_dice_path, "w") as f:
            json.dump(best_dice_dict, f, indent=4)

      return float(loss), len(self.val_dataloader), {"iou": iou, "dice": dice}



class FlowerClientFedREP(fl.client.NumPyClient):
    def __init__(self, client_id, train_dataloader, val_dataloader, input_channels, num_classes, output_dir, random_seed=42):
        """
        Creates a FlowerClient object to simulate a client.
        
        Arguments:
        train_dataloader (DataLoader): DataLoader for training data on a single client.
        val_dataloader (DataLoader): DataLoader for validation data on a single client.
        input_channels (int): Number of input channels in the model.
        num_classes (int): Number of classes in the dataset.
        random_seed (int): Random seed for reproducibility.
        """
        
        super().__init__()
        self.client_id = client_id
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.in_channels = input_channels
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.model = UnetFedRep(
          in_channels =  self.in_channels,
          num_classes = self.num_classes
        )
        self.temporaryWeightsPath = 'temporaryWeights'
        self.headWeightsPath = os.path.join(self.temporaryWeightsPath, f'fedREPhead{self.client_id}.pth')
        self.modelWeightsPath = os.path.join(self.temporaryWeightsPath,f'fedREPserver.pth')

    def set_parameters(self, params):
        """
        Updates the model parameters using the given numpy arrays.
        
        Arguments:
        params (list): List of numpy arrays representing the model parameters.
        """
        
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        
    def get_parameters(self, config):
        """
        Returns the current model parameters as numpy arrays.
        
        Arguments:
        cfg (dict): Configuration dictionary.
        
        Returns:
        params (list): List of numpy arrays representing the model parameters.
        """
        
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_head(self):
      if os.path.exists(self.headWeightsPath):
          self.model.head.load_state_dict(torch.load(self.headWeightsPath))
      else:
          print('head parameter is not found')

    def get_head(self):
      print('Saving head weights')
      torch.save(self.model.head.state_dict(), self.headWeightsPath)

    def save_server_model(self):
      print('saving fedREP server weights')
      torch.save(self.model.state_dict(),self.modelWeightsPath)

    def copy_best_weights(self):
      temporaryWeightsPaths = glob.glob(f'temporaryWeights/*{self.client_id}.pth')
      for path in temporaryWeightsPaths:
        file_name = path.split('/')[-1]
        shutil.copy2(path, os.path.join(self.output_dir,file_name))
      if os.path.exists('temporaryWeights/fedREPserver.pth'):
        shutil.copy2('temporaryWeights/fedREPserver.pth', os.path.join(self.output_dir,'fedREPserver.pth'))
       

    def fit(self, params, cfg):
        """
        Trains model on a client using the client's data.
        
        Arguments:
        params (list): List of numpy arrays representing the model parameters.
        cfg (dict): Configuration dictionary.
        
        Returns:
        params (list): List of numpy arrays representing the updated model parameters.
        length (int): Length of the training dataloader.
        Dict (dict): Additional information (Training history) to be sent to the server.
        """
        
        self.set_parameters(params)  # Update model parameters from the server
        self.set_head()
        
        # Local training
        history = fedrep_train(
          model = self.model, 
          train_dataloader = self.train_dataloader, 
          cfg = cfg, 
          device = self.device
        )

        self.get_head()
        self.save_server_model()
        
        # Length of dataloader is for FedAVG, dict is for any additional info sent to the server
        return self.get_parameters({}), len(self.train_dataloader), {"history": history}
    
    def evaluate(self, params, cfg):
        """
        Evaluates the model on the parameters sent by the server.
        
        Arguments:
        params (list): List of numpy arrays representing the model parameters.
        cfg (dict): Configuration dictionary.
        
        Returns:
        loss (float): Average loss over the validation set.
        length (int): Length of the validation dataloader.
        eval_metrics (dict): Additional evaluation metrics (IoU, Dice) to be sent to the server.
        """
        
        self.set_parameters(params)  # Update model parameters from the server
        self.set_head()
        
        loss, iou, dice = fedrep_test(self.model, self.val_dataloader, self.device)

        best_dice_path = os.path.join(self.output_dir,'best_dice.json')
        if os.path.exists(best_dice_path):
          with open(best_dice_path, "r") as f:
            best_dice_dict = json.load(f)
        else:
          best_dice_dict = {
            '0':-1.0,
            '1':-1.0,
            '2':-1.0
          }
          with open(best_dice_path, "w") as f:
              json.dump(best_dice_dict, f, indent=4)
      
        if dice > best_dice_dict[str(self.client_id)]:
          best_dice_dict[str(self.client_id)] = dice
          self.copy_best_weights()
          with open(best_dice_path, "w") as f:
              json.dump(best_dice_dict, f, indent=4)

        return float(loss), len(self.val_dataloader), {"iou": iou, "dice": dice}


class FlowerClientFedPER(fl.client.NumPyClient):
    def __init__(self, client_id, train_dataloader, val_dataloader, input_channels, num_classes, output_dir, random_seed=42):
        """
        Creates a FlowerClient object to simulate a client.
        
        Arguments:
        train_dataloader (DataLoader): DataLoader for training data on a single client.
        val_dataloader (DataLoader): DataLoader for validation data on a single client.
        input_channels (int): Number of input channels in the model.
        num_classes (int): Number of classes in the dataset.
        random_seed (int): Random seed for reproducibility.
        """
        
        super().__init__()
        self.client_id = client_id
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.in_channels = input_channels
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.model = UnetFedPer(
          in_channels =  self.in_channels,
          num_classes = self.num_classes
        )
        self.temporaryWeightsPath = 'temporaryWeights'
        self.headWeightsPath = os.path.join(self.temporaryWeightsPath, f'fedPERhead{self.client_id}.pth')
        self.modelWeightsPath = os.path.join(self.temporaryWeightsPath,f'fedPERserver.pth')

    def set_parameters(self, params):
        """
        Updates the model parameters using the given numpy arrays.
        
        Arguments:
        params (list): List of numpy arrays representing the model parameters.
        """
        
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        
    def get_parameters(self, config):
        """
        Returns the current model parameters as numpy arrays.
        
        Arguments:
        cfg (dict): Configuration dictionary.
        
        Returns:
        params (list): List of numpy arrays representing the model parameters.
        """
        
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_head(self):
      if os.path.exists(self.headWeightsPath):
          self.model.head.load_state_dict(torch.load(self.headWeightsPath))
      else:
          print('head parameter is not found')

    def get_head(self):
      print('Saving head weights')
      torch.save(self.model.head.state_dict(), self.headWeightsPath)

    def save_server_model(self):
      print('saving fedREP server weights')
      torch.save(self.model.state_dict(),self.modelWeightsPath)

    def copy_best_weights(self):
      temporaryWeightsPaths = glob.glob(f'temporaryWeights/*{self.client_id}.pth')
      for path in temporaryWeightsPaths:
        file_name = path.split('/')[-1]
        shutil.copy2(path, os.path.join(self.output_dir,file_name))
      if os.path.exists('temporaryWeights/fedPERserver.pth'):
        shutil.copy2('temporaryWeights/fedPERserver.pth', os.path.join(self.output_dir,'fedPERserver.pth'))
       

    def fit(self, params, cfg):
        """
        Trains model on a client using the client's data.
        
        Arguments:
        params (list): List of numpy arrays representing the model parameters.
        cfg (dict): Configuration dictionary.
        
        Returns:
        params (list): List of numpy arrays representing the updated model parameters.
        length (int): Length of the training dataloader.
        Dict (dict): Additional information (Training history) to be sent to the server.
        """
        
        self.set_parameters(params)  # Update model parameters from the server
        self.set_head()

        optimizer = AdamW(self.model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
          optimizer, 
          T_0=10, 
          T_mult=2, 
          eta_min=cfg['min_lr']
        )
        
        self.model.enable_body()
        self.model.enable_head()
        
        # Local training
        history = train(
          model = self.model, 
          train_dataloader = self.train_dataloader, 
          optimizer = optimizer,
          scheduler = scheduler,
          epochs = cfg['local_epochs'],
          device = self.device
        )

        self.get_head()
        self.save_server_model()
        
        # Length of dataloader is for FedAVG, dict is for any additional info sent to the server
        return self.get_parameters({}), len(self.train_dataloader), {"history": history}
    
    def evaluate(self, params, cfg):
        """
        Evaluates the model on the parameters sent by the server.
        
        Arguments:
        params (list): List of numpy arrays representing the model parameters.
        cfg (dict): Configuration dictionary.
        
        Returns:
        loss (float): Average loss over the validation set.
        length (int): Length of the validation dataloader.
        eval_metrics (dict): Additional evaluation metrics (IoU, Dice) to be sent to the server.
        """
        
        self.set_parameters(params)  # Update model parameters from the server
        self.set_head()
        
        loss, iou, dice = test(self.model, self.val_dataloader, self.device)

        best_dice_path = os.path.join(self.output_dir,'best_dice.json')
        if os.path.exists(best_dice_path):
          with open(best_dice_path, "r") as f:
            best_dice_dict = json.load(f)
        else:
          best_dice_dict = {
            '0':-1.0,
            '1':-1.0,
            '2':-1.0
          }
          with open(best_dice_path, "w") as f:
              json.dump(best_dice_dict, f, indent=4)
      
        if dice > best_dice_dict[str(self.client_id)]:
          best_dice_dict[str(self.client_id)] = dice
          self.copy_best_weights()
          with open(best_dice_path, "w") as f:
              json.dump(best_dice_dict, f, indent=4)

        return float(loss), len(self.val_dataloader), {"iou": iou, "dice": dice}

class FlowerClientFedAVG(fl.client.NumPyClient):
    def __init__(self, client_id, train_dataloader, val_dataloader, input_channels, num_classes, output_dir, random_seed=42):
        """
        Creates a FlowerClient object to simulate a client.
        
        Arguments:
        train_dataloader (DataLoader): DataLoader for training data on a single client.
        val_dataloader (DataLoader): DataLoader for validation data on a single client.
        input_channels (int): Number of input channels in the model.
        num_classes (int): Number of classes in the dataset.
        random_seed (int): Random seed for reproducibility.
        """
        
        super().__init__()
        self.client_id = client_id
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.model = UNet(in_channels=input_channels, num_classes=num_classes, random_seed=random_seed)

    def set_parameters(self, params):
        """
        Updates the model parameters using the given numpy arrays.
        
        Arguments:
        params (list): List of numpy arrays representing the model parameters.
        """
        
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        
    def get_parameters(self, config):
        """
        Returns the current model parameters as numpy arrays.
        
        Arguments:
        cfg (dict): Configuration dictionary.
        
        Returns:
        params (list): List of numpy arrays representing the model parameters.
        """
        
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def fit(self, params, cfg):
        """
        Trains model on a client using the client's data.
        
        Arguments:
        params (list): List of numpy arrays representing the model parameters.
        cfg (dict): Configuration dictionary.
        
        Returns:
        params (list): List of numpy arrays representing the updated model parameters.
        length (int): Length of the training dataloader.
        Dict (dict): Additional information (Training history) to be sent to the server.
        """
        
        self.set_parameters(params)  # Update model parameters from the server
        
        optimizer = Adam(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["local_epochs"], eta_min=cfg["min_lr"])
        
        # Local training
        history = train(self.model, self.train_dataloader, optimizer, scheduler, cfg["local_epochs"], self.device)
        
        # Length of dataloader is for FedAVG, dict is for any additional info sent to the server
        return self.get_parameters({}), len(self.train_dataloader), {"history": history}
    
    def evaluate(self, params, cfg):
        """
        Evaluates the model on the parameters sent by the server.
        
        Arguments:
        params (list): List of numpy arrays representing the model parameters.
        cfg (dict): Configuration dictionary.
        
        Returns:
        loss (float): Average loss over the validation set.
        length (int): Length of the validation dataloader.
        eval_metrics (dict): Additional evaluation metrics (IoU, Dice) to be sent to the server.
        """
        
        self.set_parameters(params)  # Update model parameters from the server
        
        loss, iou, dice = test(self.model, self.val_dataloader, self.device)

        best_dice_path = os.path.join(self.output_dir,'best_dice.json')
        if os.path.exists(best_dice_path):
          with open(best_dice_path, "r") as f:
            best_dice_dict = json.load(f)
        else:
          best_dice_dict = {
            '0':-1.0,
            '1':-1.0,
            '2':-1.0
          }
          with open(best_dice_path, "w") as f:
              json.dump(best_dice_dict, f, indent=4)
      
        if dice > best_dice_dict[str(self.client_id)]:
          best_dice_dict[str(self.client_id)] = dice
          torch.save(self.model.state_dict(),os.path.join(self.output_dir,'fedAVG.pth'))
          with open(best_dice_path, "w") as f:
              json.dump(best_dice_dict, f, indent=4)

        return float(loss), len(self.val_dataloader), {"iou": iou, "dice": dice}
    
    
def generate_client_function(strategy, train_dataloaders, val_dataloaders, input_channels, num_classes, output_dir, random_seed):
    """
    Provides a client function that the server can evoke to spawn clients.
    
    Arguments:
    train_dataloaders (list): List of DataLoader objects for training data on multiple clients.
    val_dataloaders (list): List of DataLoader objects for validation data on multiple clients.
    input_channels (int): Number of input channels in the dataset.
    num_classes (int): Number of classes in the dataset.
    random_seed (int): Random seed for reproducibility.
    
    Returns:
    client_function (function): Function to create a FlowerClient object for a specific client.
    """
    
    def client_function(client_id):
        """
        Function to create a FlowerClient object for a specific client.
        
        Arguments:
        client_id (int): ID of the client.
        
        Returns:
        client (FlowerClient): FlowerClient object for the specified client.
        """
        if strategy == 'fedOAP':
          return FlowerClientFedOAP(
            client_id=int(client_id), 
            train_dataloader=train_dataloaders[int(client_id)], 
            val_dataloader=val_dataloaders[int(client_id)], 
            input_channels=input_channels, 
            num_classes=num_classes, 
            output_dir=output_dir,
            random_seed=random_seed
          )
        elif strategy == 'fedDP':
          return FlowerClientFedDP(
            client_id=int(client_id), 
            train_dataloader=train_dataloaders[int(client_id)], 
            val_dataloader=val_dataloaders[int(client_id)], 
            input_channels=input_channels, 
            num_classes=num_classes, 
            output_dir=output_dir,
            random_seed=random_seed
          )
        elif strategy == 'fedREP':
          return FlowerClientFedREP(
            client_id=int(client_id),
            train_dataloader=train_dataloaders[int(client_id)], 
            val_dataloader=val_dataloaders[int(client_id)], 
            input_channels=input_channels, 
            num_classes=num_classes, 
            output_dir=output_dir, 
            random_seed=random_seed
          )
        elif strategy == 'fedPER':
          return FlowerClientFedPER(
            client_id=int(client_id),
            train_dataloader=train_dataloaders[int(client_id)], 
            val_dataloader=val_dataloaders[int(client_id)], 
            input_channels=input_channels, 
            num_classes=num_classes, 
            output_dir=output_dir, 
            random_seed=random_seed
          )
        elif strategy == 'fedAVG':
          return FlowerClientFedAVG(
            client_id=int(client_id),
            train_dataloader=train_dataloaders[int(client_id)], 
            val_dataloader=val_dataloaders[int(client_id)], 
            input_channels=input_channels, 
            num_classes=num_classes, 
            output_dir=output_dir, 
            random_seed=random_seed
          )
        elif strategy == 'fedADAGRAD':
          return FlowerClientFedAVG(
            client_id=int(client_id),
            train_dataloader=train_dataloaders[int(client_id)], 
            val_dataloader=val_dataloaders[int(client_id)], 
            input_channels=input_channels, 
            num_classes=num_classes, 
            output_dir=output_dir, 
            random_seed=random_seed
          )
        else :
          print('the client function for given strategy is yet to be implemented')
          exit()
    
    return client_function