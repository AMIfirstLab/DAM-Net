import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split

import numpy as np

class DataPrep(Dataset):
  def __init__(self, path):
      
    self.total_data = np.load(path, allow_pickle=True)
    z = len(self.total_data) - 1000
    self.total_data = self.total_data[0:z]
    
    self.transform = transforms.Compose([
            transforms.ToTensor(),
  ])

  def __len__(self):
    return len(self.total_data)

  def __getitem__(self, index):
    image, label = self.total_data[index]
    image = image / np.max(image)
    image = self.transform(image)
    one_hot = torch.eye(3)[label]
    
    one_hot = torch.Tensor(one_hot)

    return image, one_hot


class ValDataPrep(Dataset):
  def __init__(self, path):

    self.total_data = np.load(path, allow_pickle=True)
    z = len(self.total_data) - 1000
    self.total_data = self.total_data[z:len(self.total_data)]
    
    self.transform = transforms.Compose([
            transforms.ToTensor(),
  ])

  def __len__(self):
    return len(self.total_data)

  def __getitem__(self, index):
    image, label = self.total_data[index]
    image = image / np.max(image)
    image = self.transform(image)
    one_hot = torch.eye(3)[label]
    
    one_hot = torch.Tensor(one_hot)
    return image, one_hot