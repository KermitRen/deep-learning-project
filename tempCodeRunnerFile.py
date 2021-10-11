import torch
import numpy as np
from datasets import AnimalImageDataset
from torch.utils.data import DataLoader

#Load Data
animal_dataset = AnimalImageDataset(img_dir = 'data/train/cat')
loader = DataLoader(dataset = animal_dataset, batch_size = 64, shuffle = True)

train_features = next(iter(loader))
print(5)
print(train_features.size())
