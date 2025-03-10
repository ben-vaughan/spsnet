import os
from PIL import Image
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SpectrogramDataset(Dataset):
  def __init__(self, data, img_dir, transform=None):
    self.annotations = data
    self.img_dir = img_dir
    self.transform = transform
  
  def __len__(self):
    return len(self.annotations)
  
  def __getitem__(self, idx):
    img_name = os.path.join(self.img_dir, self.annotations.iloc[idx]['image_name'])
    image = Image.open(img_name).convert("RGB")
    labels = self.annotations.iloc[idx]['label']
    labels = torch.tensor(labels).long()
    if self.transform:
      image = self.transform(image)
    return image, labels

def getDataset():
    transform_mnist = transforms.Compose([
      transforms.Resize((32, 32)),
      transforms.ToTensor(),
    ])
    
    train_data = pd.read_csv('../data/labels_train.csv')
    test_data = pd.read_csv('../data/labels_test.csv')

    print(f"Train data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")

    trainset = SpectrogramDataset(train_data, '../data/processed/train/', transform=transform_mnist)
    testset = SpectrogramDataset(test_data, '../data/processed/test/', transform=transform_mnist)

    print("\nTest data distribution:")
    test_dist = test_data['label'].value_counts(normalize=True).sort_index() * 100
    print(test_dist.to_string(header=False, float_format='{:,.2f}%'.format))

    num_classes = 5
    inputs = 3
   
    return trainset, testset, inputs, num_classes

def getDataloader(trainset, testset, valid_size, batch_size, num_workers):
    print(f"Initial trainset size: {len(trainset)}")
    print(f"Initial testset size: {len(testset)}")

    # get unique patients
    patients = trainset.annotations['image_name'].str[12:18].unique()
    print(patients)
    
    num_patients = len(patients)
    num_valid_patients = max(1, int(valid_size * num_patients))

    print(f"Total number of patients: {num_patients}")
    print(f"Number of patients for validation: {num_valid_patients}")

    # randomly select patients for validation
    np.random.shuffle(patients)
    valid_patients = patients[:num_valid_patients]

    print(f"First few validation patients: {valid_patients[:5]}")

    # split data
    train_data = trainset.annotations[~trainset.annotations['image_name'].str[12:18].isin(valid_patients)]
    valid_data = trainset.annotations[trainset.annotations['image_name'].str[12:18].isin(valid_patients)]

    print(f"Size of train_data: {len(train_data)}")
    print(f"Size of valid_data: {len(valid_data)}")

    # create new datasets
    train_subset = SpectrogramDataset(train_data, trainset.img_dir, trainset.transform)
    valid_subset = SpectrogramDataset(valid_data, trainset.img_dir, trainset.transform)

    print(f"Train subset size: {len(train_subset)}")
    print(f"Valid subset size: {len(valid_subset)}")

    # create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, valid_loader, test_loader

