import torch
from torch import optim
import torch.nn.functional as F
import visualizing as Vis
import datasets as Data
import models as M
import utility as U

#Setup Variables
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
img_size = (256, 256)
bs = 64
epochs = 3
learning_rate = 0.01

#Load Data
starting_image = Data.getLion(image_size = img_size)
ending_image = Data.getTiger(image_size = img_size)

#Setup Base Model
model = M.AutoEncoder()
model.to(dev)
opt = optim.Adam(model.parameters(), lr = learning_rate)
U.load_model_state(model, opt, 'trained_models/DenoisingAE.pt')

#Setup DeepDream Model
DDModel = M.DeepDream(starting_image, ending_image, model)
DDOpt = optim.Adam(DDModel.parameters(), lr = learning_rate)

#Train Model
history = U.fit_DeepDream(DDModel, F.mse_loss, DDOpt, 200)
historyAE = model(history)
Vis.displayImages(history, columMax=10)
Vis.displayImages(historyAE, columMax=10)

#To do:
# - Clean models file
# - Variational autoencoder
# - Choose architecture for standard ae
# - Better saving system (Include image_size & Loss)
