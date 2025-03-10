import data

import vbll
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import multiprocessing
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logging/prob/complexified/baseline")

class Deterministic(nn.Module):
  def __init__(self, cfg):
    super(Deterministic, self).__init__()
    self.cfg = cfg
    self.params = nn.ModuleDict({
      'in_layer': nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),  # 30x30x32           L1
        nn.ReLU(),
      ),
      'core': nn.ModuleList([
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),   # 28x28x32         L3
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),   # 14x14x32         L5
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),   # 12x12x48         L7
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),   # 6x6x64           L9
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),   # 4x4x64          L11
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=0),  # 1x1x128         L13
        nn.ReLU(),
      ]),
      'out_layer': nn.Sequential(
        nn.Flatten(),
        nn.Linear(1 * 1 * 128, self.cfg.OUT_FEATURES)
      )
    })

  def forward(self, x):
    x = self.params['in_layer'](x)
    for layer in self.params['core']:
      x = layer(x)    
    return F.log_softmax(self.params['out_layer'](x), dim=-1)

class Probabilistic(nn.Module):
  def __init__(self, cfg):
    super(Probabilistic, self).__init__()
    self.cfg = cfg
    self.params = nn.ModuleDict({
      'in_layer': nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),  # 30x30x32        L1
        nn.ReLU(),
      ),
      'core': nn.ModuleList([
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),   # 28x28x32        L3
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),   # 14x14x32        L5
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),   # 12x12x48        L7
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),   # 6x6x64          L9
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),   # 4x4x64          L11
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=0),  # 1x1x128         L13
        nn.ReLU(),
      ]),
      'out_layer': nn.Sequential(
        nn.Flatten(),
        vbll.DiscClassification(1 * 1 * 128, self.cfg.OUT_FEATURES, self.cfg.REG_WEIGHT, parameterization=self.cfg.PARAM, return_ood=self.cfg.RETURN_OOD, prior_scale=self.cfg.PRIOR_SCALE)
      )
    })

  def forward(self, x):
    x = self.params['in_layer'](x)
    for layer in self.params['core']:
      x = layer(x)
    return self.params['out_layer'](x)

def save_model(model, filepath):
  torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': model.cfg,
  }, filepath)

def load_model(filepath, device, type='vbll'):
  checkpoint = torch.load(filepath, map_location=device)
  if type == 'vbll':
    loaded_model = Probabilistic(checkpoint['model_config'])
  else:
    loaded_model = Deterministic(checkpoint['model_config'])
  loaded_model.load_state_dict(checkpoint['model_state_dict'])
  loaded_model.to(device)
  return loaded_model

def eval_acc(preds, y):
  map_preds = torch.argmax(preds, dim=1)
  return (map_preds == y).float().mean()

def eval_ood(model, ind_dataloader, ood_dataloader, VBLL=False):
  ind_preds = []
  ood_preds = []

  def get_score(out):
    if VBLL:
      score = out.ood_scores.detach().cpu().numpy()
    else:
      score = torch.max(out, dim=-1)[0].detach().cpu().numpy()
    return score

  model.eval() 
  with torch.no_grad():
    for x, _ in ind_dataloader:
      x = x.to(device)
      out = model(x)
      ind_preds.extend(get_score(out))

    for x, _ in ood_dataloader:
      x = x.to(device)
      out = model(x)
      ood_preds.extend(get_score(out))

  ind_preds = np.array(ind_preds)
  ood_preds = np.array(ood_preds)

  labels = np.concatenate([np.ones(len(ind_preds)), np.zeros(len(ood_preds))])
  scores = np.concatenate([ind_preds, ood_preds])

  fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
  return metrics.auc(fpr, tpr)

def train(model, train_cfg):
  if train_cfg.VBLL:
    # for VBLL models, set weight decay to zero on last layer
    param_list = [
      {'params': model.params.in_layer.parameters(), 'weight_decay': train_cfg.WD},
      {'params': model.params.core.parameters(), 'weight_decay': train_cfg.WD},
      {'params': model.params.out_layer.parameters(), 'weight_decay': 0.}
    ]
  else:
    param_list = model.parameters()
    loss_fn = nn.CrossEntropyLoss() # for deterministic model

  optimizer = train_cfg.OPT(param_list, lr=train_cfg.LR, weight_decay=train_cfg.WD)

  dataloader = train_loader
  val_dataloader = valid_loader
  ood_dataloader = test_loader

  output_metrics = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': []
  }

  for epoch in range(train_cfg.NUM_EPOCHS):
    model.train()
    running_loss = []
    running_acc = []
    train_per_class_correct = {i: 0 for i in range(5)}
    train_per_class_total = {i: 0 for i in range(5)}

    for train_step, data in enumerate(dataloader):
      optimizer.zero_grad()
      x = data[0].to(device)
      y = data[1].to(device)

      out = model(x)                       
      if train_cfg.VBLL:
        loss = out.train_loss_fn(y)          
        probs = out.predictive.probs
        acc = eval_acc(probs, y).item()
      else:
        loss = loss_fn(out, y)
        probs = torch.exp(out)  # convert log_softmax to probabilities
        acc = eval_acc(out, y).item()

      running_loss.append(loss.item())
      running_acc.append(acc)

      _, predicted = torch.max(probs, dim=1)
      for i in range(5):
        train_per_class_correct[i] += (predicted[y == i] == y[y == i]).sum().item()
        train_per_class_total[i] += (y == i).sum().item()

      loss.backward()
      optimizer.step()

      if train_step % 100 == 0: 
        print(f'Epoch {epoch}, Batch {train_step}/{len(dataloader)}, Loss: {loss.item():.4f}, Acc: {acc:.4f}')

    train_per_class_acc = {c: train_per_class_correct[c] / train_per_class_total[c] if train_per_class_total[c] > 0 else 0 for c in range(5)}
    for class_idx, accuracy in train_per_class_acc.items():
        writer.add_scalar(f'Class {class_idx}/Training Accuracy', accuracy, epoch)

    writer.add_scalar('Overall/Training Accuracy', np.mean(running_acc), epoch)
    writer.add_scalar('Loss/Training', np.mean(running_loss), epoch)

    if epoch % train_cfg.VAL_FREQ == 0:
      running_val_loss = []
      running_val_acc = []
      val_per_class_correct = {i: 0 for i in range(5)}
      val_per_class_total = {i: 0 for i in range(5)}

      with torch.no_grad():
        model.eval()
        for test_step, data in enumerate(val_dataloader):
          x = data[0].to(device)
          y = data[1].to(device)

          out = model(x)
          if train_cfg.VBLL:
            loss = out.val_loss_fn(y)
            probs = out.predictive.probs
            acc = eval_acc(probs, y).item()
          else:
            loss = loss_fn(out, y)
            acc = eval_acc(out, y).item()
            probs = torch.exp(out) 

          running_val_loss.append(loss.item())
          running_val_acc.append(acc)

          _, predicted = torch.max(probs, dim=1)
          for i in range(5):
            val_per_class_correct[i] += (predicted[y == i] == y[y == i]).sum().item()
            val_per_class_total[i] += (y == i).sum().item()

        output_metrics['val_loss'].append(np.mean(running_val_loss))
        output_metrics['val_acc'].append(np.mean(running_val_acc))
        
      val_per_class_acc = {c: val_per_class_correct[c] / val_per_class_total[c] if val_per_class_total[c] > 0 else 0 for c in range(5)}

      for class_idx, accuracy in val_per_class_acc.items():
        writer.add_scalar(f'Class {class_idx}/Validation Accuracy', accuracy, epoch)
      writer.add_scalar('Overall/Validation Accuracy', np.mean(running_val_acc), epoch)
      writer.add_scalar('Loss/Validation', np.mean(running_val_loss), epoch)

      print('Epoch: {:2d}, train loss: {:4.4f}, train acc: {:4.4f}'.format(epoch, np.mean(running_loss), np.mean(running_acc)))
      print('Epoch: {:2d}, valid loss: {:4.4f}, val acc: {:4.4f}'.format(epoch, np.mean(np.mean(running_val_loss)), np.mean(running_val_acc)))
      print(f"Training per-class accuracies at the end of Epoch {epoch+1}: {train_per_class_acc}")
      print(f"Validation per-class accuracies at the end of Epoch {epoch+1}: {val_per_class_acc}")

  return output_metrics

if __name__ == '__main__':
  multiprocessing.freeze_support()
  device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

  trainset, testset, inputs, outputs = data.getDataset()
  train_loader, valid_loader, test_loader = data.getDataloader(trainset, testset, 0.2, 64, 4)
  outputs = {}

  class train_cfg:
    NUM_EPOCHS = 20
    BATCH_SIZE = 64
    LR = 1e-4
    WD = 1e-3
    OPT = torch.optim.AdamW
    CLIP_VAL = 1
    VAL_FREQ = 1
    VBLL = True

  class cfg:
    IN_FEATURES = 3
    OUT_FEATURES = 5
    REG_WEIGHT = 1./trainset.__len__()
    PARAM = 'diagonal'
    RETURN_OOD = True
    PRIOR_SCALE = 1.

  all_labels = []
  for _, labels in test_loader:
      all_labels.extend(labels.numpy())

  prob_model = Probabilistic(cfg()).to(device)
  outputs['prob'] = train(prob_model, train_cfg())
  
  # class train_cfg:
  #   NUM_EPOCHS = 20
  #   BATCH_SIZE = 64
  #   LR = 1e-4
  #   WD = 1e-3
  #   OPT = torch.optim.AdamW
  #   CLIP_VAL = 1
  #   VAL_FREQ = 1
  #   VBLL = False

  # det_model = Deterministic(cfg()).to(device)
  # outputs['det'] = train(det_model, train_cfg())

  save_model(prob_model, 'prob_model.pth')
  # save_model(det_model, 'det_model.pth')
