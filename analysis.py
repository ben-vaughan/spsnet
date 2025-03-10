import data
import vbll
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

class Deterministic(nn.Module):
  def __init__(self, cfg):
    super(Deterministic, self).__init__()
    self.cfg = cfg
    self.params = nn.ModuleDict({
      'in_layer': nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),  # 30x30x32        L1
        nn.ReLU(),
      ),
      'core': nn.ModuleList([
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),   # 28x28x32        L3
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),   # 14x14x32        L5
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),   # 12x12x48        L7
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),   # 6x6x64          L9
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),   # 4x4x64          L11
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=96, kernel_size=4, stride=2, padding=0),  # 1x1x128         L13
        nn.ReLU(),
      ]),
      'out_layer': nn.Sequential(
        nn.Flatten(),
        nn.Linear(1 * 1 * 96, self.cfg.OUT_FEATURES)
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
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),   # 14x14x32        L5
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),   # 12x12x48        L7
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),   # 6x6x64          L9
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),   # 4x4x64          L11
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=96, kernel_size=4, stride=2, padding=0),  # 1x1x128         L13
        nn.ReLU(),
      ]),
      'out_layer': nn.Sequential(
        nn.Flatten(),
        vbll.DiscClassification(1 * 1 * 96, self.cfg.OUT_FEATURES, self.cfg.REG_WEIGHT, parameterization=self.cfg.PARAM, return_ood=self.cfg.RETURN_OOD, prior_scale=self.cfg.PRIOR_SCALE)
      )
    })

  def forward(self, x):
    x = self.params['in_layer'](x)
    for layer in self.params['core']:
      x = layer(x)
    return self.params['out_layer'](x)

def load_model(filepath, device, type='vbll'):
  checkpoint = torch.load(filepath, map_location=device)
  if type == 'vbll':
    loaded_model = Probabilistic(checkpoint['model_config'])
  else:
    loaded_model = Deterministic(checkpoint['model_config'])
  loaded_model.load_state_dict(checkpoint['model_state_dict'])
  loaded_model.to(device)
  return loaded_model

def compute_metrics(model, loader, num_classes, device, is_vbll=False):
  model.eval()
  all_probs = []
  all_preds = []
  all_targets = []
  all_entropy = []

  with torch.no_grad():
    for inputs, targets in loader:
      inputs = inputs.to(device)
      outputs = model(inputs)
      if is_vbll:
        probs = outputs.predictive.probs
      else:
        probs = torch.exp(outputs)
      
      preds = probs.argmax(dim=1)
      
      all_probs.append(probs.cpu().numpy())
      all_preds.append(preds.cpu().numpy())
      all_targets.append(targets.numpy())
      
      entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
      all_entropy.append(entropy.cpu().numpy())

  all_probs = np.concatenate(all_probs)
  all_preds = np.concatenate(all_preds)
  all_targets = np.concatenate(all_targets)
  all_entropy = np.concatenate(all_entropy)

  return all_probs, all_preds, all_targets, all_entropy

def plot_calibration_curve(y_true, y_prob, title="calibration_curve"):
  plt.figure(figsize=(10, 4))
  
  for i in range(y_prob.shape[1]):
    prob_true, prob_pred = calibration_curve(y_true == i, y_prob[:, i], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=f'Class {i}')
  
  plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
  plt.xlabel("Mean predicted probability")
  plt.ylabel("Fraction of positives")
  plt.title(title)
  plt.legend()
  plt.savefig(f'metrics/{title.lower().replace(" ", "_")}.png')
  plt.close()

def plot_entropy_distribution(entropy, targets, num_classes):
  for i in range(num_classes):
    plt.figure(figsize=(10, 6))
    class_entropy = entropy[targets == i]
    plt.hist(class_entropy, bins=50, alpha=0.7)
    plt.title(f"Distribution of Predictive Entropy for Class {i}")
    plt.xlabel("Predictive Entropy")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f'metrics/entropy_distribution_class_{i}.png')
    plt.close()

  plt.figure(figsize=(10, 6))
  plt.hist([entropy[targets == i] for i in range(num_classes)], bins=50, alpha=0.7, label=[f'Class {i}' for i in range(num_classes)])
  plt.title("Distribution of Predictive Entropy per Class")
  plt.xlabel("Predictive Entropy")
  plt.ylabel("Count")
  plt.legend()
  plt.tight_layout()
  plt.savefig('metrics/entropy_distribution_overall.png')
  plt.close()

def brier_score_per_class(y_true, y_prob):
  num_classes = y_prob.shape[1]
  brier_scores = {}
  for i in range(num_classes):
    true_binary = (y_true == i).astype(int) 
    prob_i = y_prob[:, i]  
    brier_score_i = np.mean((prob_i - true_binary) ** 2)
    brier_scores[f'Class {i}'] = brier_score_i
  return brier_scores

def calculate_per_class_kappa(y_true, y_pred, num_classes):
  kappa_scores = {}
  for i in range(num_classes):
    y_true_binary = (y_true == i).astype(int)
    y_pred_binary = (y_pred == i).astype(int)
    
    kappa = cohen_kappa_score(y_true_binary, y_pred_binary)
    kappa_scores[f'Class_{i}'] = kappa
  return kappa_scores

def compute_class_metrics(y_true, y_pred, y_prob):
  num_classes = y_prob.shape[1]
  metrics = {}

  for i in range(num_classes):
    metrics[f'Class_{i}'] = {
      'Accuracy': (y_true == i) == (y_pred == i),
      'F1 Score': f1_score(y_true == i, y_pred == i),
      'Precision': precision_recall_curve(y_true == i, y_prob[:, i])[0],
      'Recall': precision_recall_curve(y_true == i, y_prob[:, i])[1],
      'Average Precision': average_precision_score(y_true == i, y_prob[:, i])
    }
  return metrics

def plot_entropy_distribution_comparison(det_entropy, prob_entropy, det_targets, prob_targets, num_classes):
  for i in range(num_classes):
    plt.figure(figsize=(10, 6))
    
    det_class_entropy = det_entropy[det_targets == i]
    prob_class_entropy = prob_entropy[prob_targets == i]
    
    plt.hist(det_class_entropy, bins=60, alpha=0.5, label='Deterministic', density=True)
    plt.hist(prob_class_entropy, bins=60, alpha=0.5, label='Probabilistic', density=True)
    
    plt.title(f"Distribution of Predictive Entropy for Class {i}")
    plt.xlabel("Predictive Entropy")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'metrics/entropy_distribution_comparison_class_{i}.png')
    plt.close()

  # Overall dist
  plt.figure(figsize=(10, 6))
  plt.hist(det_entropy, bins=50, alpha=0.5, label='Deterministic', density=True)
  plt.hist(prob_entropy, bins=50, alpha=0.5, label='Probabilistic', density=True)
  plt.title("Overall Distribution of Predictive Entropy")
  plt.xlabel("Predictive Entropy")
  plt.ylabel("Density")
  plt.legend()
  plt.tight_layout()
  plt.savefig('metrics/entropy_distribution_comparison_overall.png')
  plt.close()

def compute_per_class_ece(probs, labels, n_bins=10, n_classes=5):
  """Compute Expected Calibration Error (ECE) for each class."""
  per_class_ece = {}

  for class_idx in range(n_classes):
    binary_labels = (labels == class_idx).astype(int)
    class_probs = probs[:, class_idx]

    ece = 0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    for bin in range(n_bins):
      bin_start, bin_end = bin_boundaries[bin:bin+2]
      bin_mask = np.logical_and(class_probs > bin_start, class_probs <= bin_end)
      
      if np.sum(bin_mask) > 0:
        bin_accuracy = np.mean(binary_labels[bin_mask])
        bin_confidence = np.mean(class_probs[bin_mask])
        ece += np.abs(bin_accuracy - bin_confidence) * np.sum(bin_mask) / len(labels)

    per_class_ece[f'Class_{class_idx}'] = ece

  return per_class_ece


def compute_dummy_metrics(test_loader, num_classes=5, strategy='stratified'):
    all_targets = []
    for _, targets in test_loader:
      all_targets.extend(targets.numpy())
    y_test = np.array(all_targets)

    dummy_clf = DummyClassifier(strategy=strategy, random_state=42)
    dummy_clf.fit(np.zeros((len(y_test), 1)), y_test)

    # Get predictions and probabilities
    dummy_preds = dummy_clf.predict(np.zeros((len(y_test), 1)))
    dummy_probs = dummy_clf.predict_proba(np.zeros((len(y_test), 1)))

    accuracy = accuracy_score(y_test, dummy_preds)
    kappa = cohen_kappa_score(y_test, dummy_preds)
    
    f1_per_class = {f'Class_{i}': f1_score(y_test == i, dummy_preds == i) for i in range(num_classes)}
    kappa_per_class = {f'Class_{i}': cohen_kappa_score(y_test == i, dummy_preds == i) for i in range(num_classes)}
    
    per_class_ece = compute_per_class_ece(dummy_probs, y_test, n_bins=10, n_classes=num_classes)
    avg_ece = sum(per_class_ece.values()) / len(per_class_ece)

    brier_scores = brier_score_per_class(y_test, dummy_probs)

    return dummy_preds, dummy_probs, y_test, accuracy, kappa, f1_per_class, kappa_per_class, per_class_ece, avg_ece, brier_scores

if __name__ == '__main__':
  device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

  trainset, testset, inputs, outputs = data.getDataset()
  train_loader, valid_loader, test_loader = data.getDataloader(trainset, testset, 0.2, 64, 4)

  class cfg:
    IN_FEATURES = 3
    OUT_FEATURES = 5
    REG_WEIGHT = 1./trainset.__len__()
    PARAM = 'diagonal'
    RETURN_OOD = True
    PRIOR_SCALE = 1.

  prob_model_path = 'models/baseline/prob_model.pth'
  det_model_path = 'models/baseline/det_model.pth'
  num_classes = 5

  prob_model = load_model(prob_model_path, device)
  det_model = load_model(det_model_path, device, type='det')    

  det_probs, det_preds, det_targets, det_entropy = compute_metrics(det_model, test_loader, num_classes=5, device=device, is_vbll=False)
  prob_probs, prob_preds, prob_targets, prob_entropy = compute_metrics(prob_model, test_loader, num_classes=5, device=device, is_vbll=True)