import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from datasets import AnimalImageDataset
from torch.utils.data import DataLoader

#Load Data
animal_dataset = AnimalImageDataset(img_dir = 'data/train', size = (256,256))
loader = DataLoader(dataset = animal_dataset, batch_size = 64, shuffle = True)

train_features = next(iter(loader))
print(train_features.size())

img = train_features[0]

grid_img = torchvision.utils.make_grid(train_features[0:10], nrow=5)

plt.imshow(grid_img.permute(1, 2, 0))
plt.show()