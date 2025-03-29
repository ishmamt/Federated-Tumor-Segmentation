import os
import flwr as fl
import torch
from collections import OrderedDict
from torch.optim import Adam
from torch.optim import lr_scheduler

from models.unetWithModifications import SharedUnet,PersonalizedUnet,\
SharedDown,QGenerator,SharedUpWithAttn
# from train import train, test
from trainWithModifications import trainWithAdapter,testWithAdapter,\
trainWithAttention, testWithAttention


class FlowerClientWithAttention(fl.client.NumPyClient):
  def __init__(self, client_id, train_dataloader, val_dataloader, input_channels, num_classes, random_seed=42):
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
      
      self.modelDown = SharedDown(
        in_channels=input_channels, 
        random_seed=random_seed
      )
      outputChannelsOfModelDown = self.modelDown.output_channels
      self.qGenerator = QGenerator(
        in_channels = outputChannelsOfModelDown,
        random_seed = random_seed
      )
      self.modelUp = SharedUpWithAttn(
        in_channels = outputChannelsOfModelDown,
        num_classes = num_classes,
        random_seed = random_seed
      )
      ## Let's hardcode the model weight path for now
      self.q_weight_path = \
      f'/content/drive/MyDrive/UFF/Federated-Tumor-Segmentation/q_weight/model{client_id}.pth'
      # if os.path.exists(self.adapter_weight_path):
      #   os.remove(self.adapter_weight_path)

  def set_parameters_qGenerator(self):
    # Check if the file exists
    if os.path.exists(self.q_weight_path):
        print("Loading saved qGenerator weights...")
        self.qGenerator.load_state_dict(torch.load(self.q_weight_path))
    else:
        print("No saved weights found, skipping loading qGenerator.")

  def set_parameters(self, params):
      """
      Updates the model parameters using the given numpy arrays.
      
      Arguments:
      params (list): List of numpy arrays representing the model parameters.
      """
      
      num_params_down = len(list(self.modelDown.parameters()))
      params_down = params[:num_params_down]  # First half for Model A
      params_up = params[num_params_down:]  # Second half for Model B

      # Load into Model A
      for param, model_param in zip(params_down, self.modelDown.parameters()):
          model_param.data = torch.tensor(param)

      # Load into Model B
      for param, model_param in zip(params_up, self.modelUp.parameters()):
          model_param.data = torch.tensor(param)

  def get_parameters_qGenerator(self):
      torch.save(self.qGenerator.state_dict(),self.q_weight_path)

  def get_parameters(self, config):
      """
      Returns the current model parameters as numpy arrays.
      
      Arguments:
      cfg (dict): Configuration dictionary.
      
      Returns:
      params (list): List of numpy arrays representing the model parameters.
      """
      
      # params_down = [p.cpu().clone().detach().numpy() for p in self.modelDown.parameters()]
      # params_up = [p.cpu().clone().detach().numpy() for p in self.modelUp.parameters()]
      # return params_down + params_up  # Flatten into one list
      return [val.cpu().numpy() for _, val in self.modelDown.state_dict().items()]+\
      [val.cpu().numpy() for _, val in self.modelUp.state_dict().items()]
  
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
      self.set_parameters_qGenerator()

      optimizer = Adam(
          list(self.modelDown.parameters())+
          list(self.qGenerator.parameters())+
          list(self.modelUp.parameters()), 
          lr=cfg["lr"], 
          weight_decay=cfg["weight_decay"]
        )
      scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["local_epochs"], eta_min=cfg["min_lr"])
      
      # Local training
      history = trainWithAttention(
          (self.modelDown,self.qGenerator,self.modelUp), 
          self.train_dataloader, 
          optimizer, 
          scheduler, 
          cfg["local_epochs"], 
          self.device
        )
      
      self.get_parameters_qGenerator()

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
      
      loss, iou, dice = testWithAttention(
          (self.modelDown,self.qGenerator,self.modelUp), 
          self.val_dataloader, 
          self.device
        )
      
      return float(loss), len(self.val_dataloader), {"iou": iou, "dice": dice}


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, train_dataloader, val_dataloader, input_channels, num_classes, random_seed=42):
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
        
        self.model = SharedUnet(in_channels=input_channels, num_classes=num_classes, random_seed=random_seed)
        self.adapter = PersonalizedUnet(
          in_channels = self.model.featureSizeForAdapter,
          num_classes = num_classes,
          random_seed = random_seed
        )
        ## Let's hardcode the model weight path for now
        self.adapter_weight_path = \
        f'/content/drive/MyDrive/UFF/Federated-Tumor-Segmentation/adapter_weight/model{self.client_id}.pth'
        # if os.path.exists(self.adapter_weight_path):
        #   os.remove(self.adapter_weight_path)

    def set_parameters_adapter(self):
      # Check if the file exists
      if os.path.exists(self.adapter_weight_path):
          print("Loading saved adapter weights...")
          self.adapter.load_state_dict(torch.load(self.adapter_weight_path))
      else:
          print("No saved weights found, skipping loading.")

    def set_parameters(self, params):
        """
        Updates the model parameters using the given numpy arrays.
        
        Arguments:
        params (list): List of numpy arrays representing the model parameters.
        """
        
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters_adapter(self):
        torch.save(self.adapter.state_dict(),self.adapter_weight_path)

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
        self.set_parameters_adapter()

        optimizer = Adam(
            list(self.model.parameters())+list(self.adapter.parameters()), 
            lr=cfg["lr"], 
            weight_decay=cfg["weight_decay"]
          )
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["local_epochs"], eta_min=cfg["min_lr"])
        
        # Local training
        history = trainWithAdapter(
            self.model, 
            self.adapter, 
            self.train_dataloader, 
            optimizer, 
            scheduler, 
            cfg["local_epochs"], 
            self.device
          )
        
        self.get_parameters_adapter()

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
        
        loss, iou, dice = testWithAdapter(
            self.model, 
            self.adapter, 
            self.val_dataloader, 
            self.device
          )
        
        return float(loss), len(self.val_dataloader), {"iou": iou, "dice": dice}
    
    
def generate_client_function(train_dataloaders, val_dataloaders, input_channels, num_classes, random_seed):
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
        
        return FlowerClientWithAttention(int(client_id), train_dataloaders[int(client_id)], val_dataloaders[int(client_id)], 
                            input_channels, num_classes, random_seed)
    
    return client_function