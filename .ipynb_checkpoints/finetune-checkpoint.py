import torch
from tqdm import tqdm
from flwr.common.logger import log
from logging import INFO
import yaml
import json
import os
from copy import deepcopy
from torch.optim import AdamW,lr_scheduler

from metrics import iou_dice_score
from loss import BCEDiceLoss
from utils import AvgMeter,weighted_bce_loss,compute_pred_uncertainty
from models.fedOAP import UNetWithCrossAttention
from models.fedDP import UNetWithAttention
from models.fedREP import UnetFedRep
from datasets.dataset import prepare_datasets, load_datasets

USE_FEDOAP_FINETUNE = True
NOT_USE_FEDOAP_PERSONALISED = False

class FinetuneFedOAP():
  def __init__(self,
      train_dataloaders,
      val_dataloaders,
      test_dataloaders,
      config_fit,
      device,
      output_dir,
      epochs=10,
      val_per_epoch=1,
      in_channels=3, 
      num_classes=1,
      run=1,
      error_threshold=0.75,
      noisy_loss_threshold=0.1
    ):

    self.train_dataloaders = train_dataloaders
    self.val_dataloaders = val_dataloaders
    self.test_dataloaders = test_dataloaders

    self.num_clients = len(train_dataloaders)
    self.in_channels = in_channels
    self.num_classes = num_classes 
    self.config_fit = config_fit
    self.device = device
    self.output_dir = output_dir
    self.epochs = epochs
    self.run = run
    self.val_per_epoch = val_per_epoch
    self.error_threshold = error_threshold
    self.noisy_loss_threshold = noisy_loss_threshold
    self.clients = self.init_models()
    self.optimizers, self.schedulers = self.init_opt_sch()

  def init_models(self):
    clients = []

    for idx in range(self.num_clients):
      client = UNetWithCrossAttention(
        in_channels=self.in_channels,
        num_classes=self.num_classes
      )

      serverPath = os.path.join(self.output_dir,'fedOAPserver.pth')
      if os.path.exists(serverPath):
        log(INFO, f'loading server weights for client {idx}')
        client.load_state_dict(torch.load(serverPath))
      else:
        log(INFO, f"server weights for client {idx} is not found")

      if NOT_USE_FEDOAP_PERSONALISED:
        clients.append(client)
        continue
      
      bottleneckQPath = os.path.join(self.output_dir,f'fedOAPbottleneckQ{idx}.pth')
      if os.path.exists(bottleneckQPath):
        log(INFO, f'loading bottleneckQ weights for client {idx}')
        client.bottleneckQ.load_state_dict(torch.load(bottleneckQPath))
      else:
        log(INFO, f"bottleneckQ weights for client {idx} is not found")
      
      queryPath = os.path.join(self.output_dir,f'fedOAPquery{idx}.pth')
      if os.path.exists(queryPath):
        log(INFO, f"loading query weights for client {idx}")
        client.cross_attn.query.load_state_dict(torch.load(queryPath))
      else:
        log(INFO, f"query weights for client {idx} is not found")
      
      adapterPath = os.path.join(self.output_dir,f'fedOAPadapter{idx}.pth')
      if os.path.exists(adapterPath):
        log(INFO, f'loading adapter weights for client {idx}')
        client.adapter.load_state_dict(torch.load(adapterPath))
      else:
        log(INFO, f'adapter qeights for client {idx} is not found')
      
      outPath = os.path.join(self.output_dir,f'fedOAPout{idx}.pth')
      if os.path.exists(outPath):
        log(INFO, f"loading outc weights for client {idx}")
        client.outc.load_state_dict(torch.load(outPath))
      else:
        log(INFO, f'outc weights for client {idx} is not found')
      
      clients.append(client)
    
    return clients
  
  def init_opt_sch(self):
    optimizers = []
    schedulers = []

    for idx in range(self.num_clients):
      optimizer = AdamW(
        self.clients[idx].parameters(), 
        lr=self.config_fit['lr'], 
        weight_decay=self.config_fit['weight_decay']
      )
      scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10, 
        T_mult=2, 
        eta_min=self.config_fit['min_lr']
      )

      optimizers.append(optimizer)
      schedulers.append(scheduler)

    return optimizers, schedulers
  
  def generate_noisy_outputs(self,outputs,masks):
    outputs_prob = torch.sigmoid(outputs)
    error_map = torch.abs(outputs_prob - masks)
    threshold = 0.5
    high_error_mask = (error_map > self.error_threshold).float()
    noise = torch.randn_like(outputs) * 0.1
    noisy_outputs = outputs + noise * high_error_mask
    return noisy_outputs
  
  def train(self):

    histories = []

    for idx in range(self.num_clients):
      criterion = BCEDiceLoss()
      self.clients[idx].train()
      self.clients[idx].to(self.device)
      
      train_avg_meters = {"loss": AvgMeter(), "iou": AvgMeter(), "dice": AvgMeter()}
      history = {"epoch": [], 
                  "lr": [],
                  "train_loss": [],
                  "train_iou": [],
                  "train_dice": []
                  }
      best_dice = -1.0
      for epoch in range(self.epochs):
          print(f"Epoch {epoch} / {self.epochs}:\n")
          loop = tqdm(self.train_dataloaders[idx])
          
          for images, masks in loop:
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizers[idx].zero_grad()
            outputs = self.clients[idx](images)

            noisy_outputs = self.generate_noisy_outputs(outputs,masks) 
            
            loss = criterion(outputs, masks)
            noisy_loss = criterion(noisy_outputs, masks)

            loss = ((1-self.noisy_loss_threshold)*loss)+(self.noisy_loss_threshold*noisy_loss)

            iou, dice = iou_dice_score(outputs, masks)
            loss.backward()
            self.optimizers[idx].step()
            
            train_avg_meters["loss"].update(loss.item())
            train_avg_meters["iou"].update(iou)
            train_avg_meters["dice"].update(dice)
            
            loop.set_postfix({"IoU": iou, "Dice": dice, "loss": loss.item()})
          
          self.schedulers[idx].step()

          if (epoch % self.val_per_epoch) == 0:
            dice = self.val(self.clients[idx],idx)
            if dice > best_dice:
              best_dice = dice
              torch.save(
                self.clients[idx].state_dict(),
                os.path.join(self.output_dir,f'fedOAPfinetuned{idx}.pth')
              )
          
          # Updating history
          history["epoch"].append(epoch)
          history["lr"].append(self.schedulers[idx].get_last_lr()[0])
          history["train_loss"].append(train_avg_meters["loss"].avg)
          history["train_iou"].append(train_avg_meters["iou"].avg)
          history["train_dice"].append(train_avg_meters["dice"].avg)
          
      histories.append(history)
    
    return histories
  
  def val(self,client,ix):
    test_avg_meters = {"loss": AvgMeter(), "iou": AvgMeter(), "dice": AvgMeter()}
    client.eval()
    client.to(self.device)
    
    with torch.no_grad():
      loop = tqdm(self.val_dataloaders[ix])
      
      for images, masks in loop:
        images, masks = images.to(self.device), masks.to(self.device)
        outputs = client(images)
        iou, dice = iou_dice_score(outputs, masks)
        test_avg_meters["dice"].update(dice)
      return test_avg_meters["dice"].avg

  def test(self):
    results = []
    self.clients = self.init_models()
    for idx in range(self.num_clients):
      criterion = BCEDiceLoss()
      test_avg_meters = {"loss": AvgMeter(), "iou": AvgMeter(), "dice": AvgMeter()}
      if USE_FEDOAP_FINETUNE:
        self.clients[idx].load_state_dict(
          torch.load(os.path.join(self.output_dir,f'fedOAPfinetuned{idx}.pth'))
        )
      self.clients[idx].eval()
      self.clients[idx].to(self.device)
      
      with torch.no_grad():
        loop = tqdm(self.test_dataloaders[idx])
        
        for images, masks in loop:
          images, masks = images.to(self.device), masks.to(self.device)
          outputs = self.clients[idx](images)
          
          iou, dice = iou_dice_score(outputs, masks)
          loss = criterion(outputs, masks).item()
          
          test_avg_meters["loss"].update(loss)
          test_avg_meters["iou"].update(iou)
          test_avg_meters["dice"].update(dice)
          
          loop.set_postfix({"IoU": iou, "Dice": dice, "loss": loss})
      
        results.append((
          test_avg_meters["loss"].avg, 
          test_avg_meters["iou"].avg, 
          test_avg_meters["dice"].avg
        ))
    
    result_dict = {
      '0':-1.0,
      '1':-1.0,
      '2':-1.0
    }
    for idx in range(self.num_clients):
        result_dict[str(idx)] = results[idx][2]
    with open(os.path.join(self.output_dir,f'results{self.run}.json'), "w") as f:
        json.dump(result_dict, f, indent=4)



class FineTuneFedDP():
  def __init__(self,
      train_dataloaders,
      val_dataloaders,
      test_dataloaders,
      config_fit,
      device,
      output_dir,
      epochs=10,
      val_per_epoch=1,
      in_channels=3, 
      num_classes=1,
      run=1
    ):
    self.train_dataloaders = train_dataloaders
    self.val_dataloaders = val_dataloaders
    self.test_dataloaders = test_dataloaders
    self.config_fit = config_fit
    self.device = device
    self.output_dir = output_dir
    self.epochs = epochs
    self.val_per_epoch = val_per_epoch
    self.in_channels = in_channels
    self.num_classes = num_classes
    self.run = run
    self.num_clients = len(train_dataloaders)
    self.clients = self.init_models()
    self.optimizers, self.schedulers = self.init_opt_sch()
    self.clientsForIgc = deepcopy(self.clients)
    for ix in range(self.num_clients):
        self.clientsForIgc[ix].to(self.device)

  def init_models(self):
    clients = []

    for idx in range(self.num_clients):
      client = UNetWithAttention(
        in_channels=self.in_channels,
        num_classes=self.num_classes
      )

      serverPath = os.path.join(self.output_dir,'fedDPserver.pth')
      if os.path.exists(serverPath):
        log(INFO, f'loading server weights for client {idx}')
        client.load_state_dict(torch.load(serverPath))
      else:
        log(INFO, f"server weights for client {idx} is not found")

      queryWeightsPath = os.path.join(self.output_dir,f'fedDPquery{idx}.pth')
      if os.path.exists(queryWeightsPath):
        log(INFO, f'loading query weights for client {idx}')
        client.attn.query.load_state_dict(torch.load(queryWeightsPath))
      else:
        log(INFO, f"query weights for client {idx} is not found")
      
      client.to(self.device)
      clients.append(client)
    
    return clients
  
  def init_opt_sch(self):
    optimizers = []
    schedulers = []

    for idx in range(self.num_clients):
      optimizer = AdamW(
        self.clients[idx].parameters(), 
        lr=self.config_fit['lr'], 
        weight_decay=self.config_fit['weight_decay']
      )
      scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10, 
        T_mult=2, 
        eta_min=self.config_fit['min_lr']
      )

      optimizers.append(optimizer)
      schedulers.append(scheduler)

    return optimizers, schedulers

  def train(self):
    histories = []
    for idx in range(self.num_clients):
      criterion = BCEDiceLoss()
      self.clients[idx].train()
      
      train_avg_meters = {"loss": AvgMeter(), "iou": AvgMeter(), "dice": AvgMeter()}
      history = {"epoch": [], 
                  "lr": [],
                  "train_loss": [],
                  "train_iou": [],
                  "train_dice": []
                  }
      best_dice = -1.0
      for epoch in range(self.epochs):
          print(f"Epoch {epoch} / {self.epochs}:\n")
          loop = tqdm(self.train_dataloaders[idx])

          for images, masks in loop:
              images, masks = images.to(self.device), masks.to(self.device)
              
              self.optimizers[idx].zero_grad()
              outputs = self.clients[idx](images)

              un_map, preds = compute_pred_uncertainty(
                net_clients=self.clientsForIgc,
                images=images
              )

              loss = criterion(outputs, masks)
              iou, dice = iou_dice_score(outputs, masks)
              bd_loss = weighted_bce_loss(outputs, masks, un_map)
              loss = loss + bd_loss * 1
      
              loss.backward()
              self.optimizers[idx].step()
              
              train_avg_meters["loss"].update(loss.item())
              train_avg_meters["iou"].update(iou)
              train_avg_meters["dice"].update(dice)
              
              loop.set_postfix({"IoU": iou, "Dice": dice, "loss": loss.item()})
          
          self.schedulers[idx].step()

          if (epoch % self.val_per_epoch) == 0:
            dice = self.val(self.clients[idx],idx)
            if dice > best_dice:
              best_dice = dice
              torch.save(
                self.clients[idx].state_dict(),
                os.path.join(self.output_dir,f'fedDPfinetuned{idx}.pth')
              )
          
          # Updating history
          history["epoch"].append(epoch)
          history["lr"].append(self.schedulers[idx].get_last_lr()[0])
          history["train_loss"].append(train_avg_meters["loss"].avg)
          history["train_iou"].append(train_avg_meters["iou"].avg)
          history["train_dice"].append(train_avg_meters["dice"].avg)
          
      histories.append(history)
    return histories

  def val(self,client,ix):
    test_avg_meters = {"loss": AvgMeter(), "iou": AvgMeter(), "dice": AvgMeter()}
    client.eval()
    client.to(self.device)
    
    with torch.no_grad():
      loop = tqdm(self.val_dataloaders[ix])
      
      for images, masks in loop:
        images, masks = images.to(self.device), masks.to(self.device)
        outputs = client(images)
        iou, dice = iou_dice_score(outputs, masks)
        test_avg_meters["dice"].update(dice)
      return test_avg_meters["dice"].avg
    
  def test(self):
    results = []
    self.clients = self.init_models()
    for idx in range(self.num_clients):
      criterion = BCEDiceLoss()
      test_avg_meters = {"loss": AvgMeter(), "iou": AvgMeter(), "dice": AvgMeter()}
      self.clients[idx].load_state_dict(
              torch.load(os.path.join(self.output_dir,f'fedDPfinetuned{idx}.pth'))
      )
      self.clients[idx].eval()
      self.clients[idx].to(self.device)
      
      with torch.no_grad():
        loop = tqdm(self.test_dataloaders[idx])
        
        for images, masks in loop:
          images, masks = images.to(self.device), masks.to(self.device)
          outputs = self.clients[idx](images)
          
          iou, dice = iou_dice_score(outputs, masks)
          loss = criterion(outputs, masks).item()
          
          test_avg_meters["loss"].update(loss)
          test_avg_meters["iou"].update(iou)
          test_avg_meters["dice"].update(dice)
          
          loop.set_postfix({"IoU": iou, "Dice": dice, "loss": loss})
      
        results.append((
          test_avg_meters["loss"].avg, 
          test_avg_meters["iou"].avg, 
          test_avg_meters["dice"].avg
        ))
    
    result_dict = {
      '0':-1.0,
      '1':-1.0,
      '2':-1.0
    }
    for idx in range(self.num_clients):
        result_dict[str(idx)] = results[idx][2]
    with open(os.path.join(self.output_dir,f'results{self.run}.json'), "w") as f:
        json.dump(result_dict, f, indent=4)



class FineTuneFedREP():
  def __init__(self,
      train_dataloaders,
      val_dataloaders,
      test_dataloaders,
      config_fit,
      device,
      output_dir,
      epochs=10,
      val_per_epoch=1,
      in_channels=3, 
      num_classes=1,
      run=1
    ):
    self.train_dataloaders = train_dataloaders
    self.val_dataloaders = val_dataloaders
    self.test_dataloaders = test_dataloaders
    self.config_fit = config_fit
    self.device = device
    self.output_dir = output_dir
    self.epochs = epochs
    self.val_per_epoch = val_per_epoch
    self.in_channels = in_channels
    self.num_classes = num_classes
    self.run = run
    self.num_clients = len(train_dataloaders)
    self.clients = self.init_models()
    self.optimizers, self.schedulers = self.init_opt_sch()
    self.clientsForIgc = deepcopy(self.clients)
    for ix in range(self.num_clients):
        self.clientsForIgc[ix].to(self.device)

  def init_models(self):
    clients = []

    for idx in range(self.num_clients):
      client = UnetFedRep(
        in_channels=self.in_channels,
        num_classes=self.num_classes
      )

      serverPath = os.path.join(self.output_dir,'fedREPserver.pth')
      if os.path.exists(serverPath):
        log(INFO, f'loading server weights for client {idx}')
        client.load_state_dict(torch.load(serverPath))
      else:
        log(INFO, f"server weights for client {idx} is not found")

      headWeightsPath = os.path.join(self.output_dir,f'fedREPhead{idx}.pth')
      if os.path.exists(headWeightsPath):
        log(INFO, f'loading head weights for client {idx}')
        client.head.load_state_dict(torch.load(headWeightsPath))
      else:
        log(INFO, f"head weights for client {idx} is not found")
      
      client.to(self.device)
      clients.append(client)
    
    return clients
  
  def init_opt_sch(self):
    optimizers = []
    schedulers = []

    for idx in range(self.num_clients):
      optimizer = AdamW(
        self.clients[idx].parameters(), 
        lr=self.config_fit['lr'], 
        weight_decay=self.config_fit['weight_decay']
      )
      scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10, 
        T_mult=2, 
        eta_min=self.config_fit['min_lr']
      )

      optimizers.append(optimizer)
      schedulers.append(scheduler)

    return optimizers, schedulers

  def train(self):
    histories = []
    for idx in range(self.num_clients):
      criterion = BCEDiceLoss()
      self.clients[idx].train()
      
      train_avg_meters = {"loss": AvgMeter(), "iou": AvgMeter(), "dice": AvgMeter()}
      history = {"epoch": [], 
                  "lr": [],
                  "train_loss": [],
                  "train_iou": [],
                  "train_dice": []
                  }
      best_dice = -1.0
      for epoch in range(self.epochs):
          print(f"Epoch {epoch} / {self.epochs}:\n")
          loop = tqdm(self.train_dataloaders[idx])

          for images, masks in loop:
              images, masks = images.to(self.device), masks.to(self.device)
              
              self.optimizers[idx].zero_grad()
              outputs = self.clients[idx](images)

              loss = criterion(outputs, masks)
              iou, dice = iou_dice_score(outputs, masks)
      
              loss.backward()
              self.optimizers[idx].step()
              
              train_avg_meters["loss"].update(loss.item())
              train_avg_meters["iou"].update(iou)
              train_avg_meters["dice"].update(dice)
              
              loop.set_postfix({"IoU": iou, "Dice": dice, "loss": loss.item()})
          
          self.schedulers[idx].step()

          if (epoch % self.val_per_epoch) == 0:
            dice = self.val(self.clients[idx],idx)
            if dice > best_dice:
              best_dice = dice
              torch.save(
                self.clients[idx].state_dict(),
                os.path.join(self.output_dir,f'fedREPfinetuned{idx}.pth')
              )
          
          # Updating history
          history["epoch"].append(epoch)
          history["lr"].append(self.schedulers[idx].get_last_lr()[0])
          history["train_loss"].append(train_avg_meters["loss"].avg)
          history["train_iou"].append(train_avg_meters["iou"].avg)
          history["train_dice"].append(train_avg_meters["dice"].avg)
          
      histories.append(history)
    return histories

  def val(self,client,ix):
    test_avg_meters = {"loss": AvgMeter(), "iou": AvgMeter(), "dice": AvgMeter()}
    client.eval()
    client.to(self.device)
    
    with torch.no_grad():
      loop = tqdm(self.val_dataloaders[ix])
      
      for images, masks in loop:
        images, masks = images.to(self.device), masks.to(self.device)
        outputs = client(images)
        iou, dice = iou_dice_score(outputs, masks)
        test_avg_meters["dice"].update(dice)
      return test_avg_meters["dice"].avg
    
  def test(self):
    results = []
    self.clients = self.init_models()
    for idx in range(self.num_clients):
      criterion = BCEDiceLoss()
      test_avg_meters = {"loss": AvgMeter(), "iou": AvgMeter(), "dice": AvgMeter()}
      self.clients[idx].load_state_dict(
              torch.load(os.path.join(self.output_dir,f'fedREPfinetuned{idx}.pth'))
      )
      self.clients[idx].eval()
      self.clients[idx].to(self.device)
      
      with torch.no_grad():
        loop = tqdm(self.test_dataloaders[idx])
        
        for images, masks in loop:
          images, masks = images.to(self.device), masks.to(self.device)
          outputs = self.clients[idx](images)
          
          iou, dice = iou_dice_score(outputs, masks)
          loss = criterion(outputs, masks).item()
          
          test_avg_meters["loss"].update(loss)
          test_avg_meters["iou"].update(iou)
          test_avg_meters["dice"].update(dice)
          
          loop.set_postfix({"IoU": iou, "Dice": dice, "loss": loss})
      
        results.append((
          test_avg_meters["loss"].avg, 
          test_avg_meters["iou"].avg, 
          test_avg_meters["dice"].avg
        ))
    
    result_dict = {
      '0':-1.0,
      '1':-1.0,
      '2':-1.0
    }
    for idx in range(self.num_clients):
        result_dict[str(idx)] = results[idx][2]
    with open(os.path.join(self.output_dir,f'results{self.run}.json'), "w") as f:
        json.dump(result_dict, f, indent=4)



if __name__ == '__main__':
  
  with open('conf/fedOAP.yaml', "r") as f:
    cfg = yaml.safe_load(f)

  datasets = load_datasets(cfg['dataset_dirs'], cfg['image_size'])
  log(INFO, f"Datasets loaded. Number of datasets: {len(datasets)}")
  train_dataloaders, val_dataloaders, test_dataloaders = prepare_datasets(
    datasets=datasets, 
    batch_size=cfg['batch_size'], 
    num_clients=cfg['num_clients'], 
    random_seed=cfg['random_seed'], 
    train_ratio=cfg['train_ratio'], 
    val_ratio=cfg['val_ratio']
  )

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  trainer = FinetuneFedOAP(
    train_dataloaders=train_dataloaders,
    val_dataloaders=val_dataloaders,
    test_dataloaders=test_dataloaders,
    config_fit=cfg['config_fit'],
    device=device,
    epochs=1,
    val_per_epoch=cfg['val_per_epoch'],
    in_channels=cfg['input_channels'],
    num_classes=cfg['num_classes']
  )

  trainer.train()
  trainer.test()

  del trainer
