import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import visualizing as Vis
import datasets as Data
import models as M
import utility as U

#Setup Variables
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
img_size = (256, 256)
bs = 64
epochs = 1
learning_rate = 0.001

#Load Data
animal_dataset_train = Data.AnimalImageDataset(img_dir = 'data/train',image_size = img_size, data_size=100)
animal_dataset_val = Data.AnimalImageDataset(img_dir = 'data/val', image_size = img_size)
loader_train = DataLoader(dataset = animal_dataset_train, batch_size = bs, shuffle = True)
loader_val = DataLoader(dataset = animal_dataset_val, batch_size = bs, shuffle = False)

#Setup Model
model = M.AutoEncoder()
model.to(dev)
opt = optim.Adam(model.parameters(), lr = learning_rate)

U.load_model_state(model, opt, 'trained_models/DenoisingAE.pt')

#nn.crossEntropyLoss

#Train Model
history = U.fit_AutoEncoder(loader_train, model, nn.MSELoss(), opt, epochs, loader_val, graph_loss = True)
#U.save_model_state(model, opt, 'trained_models/Classifier.pt')
Vis.displayImages(history, columMax=5)
