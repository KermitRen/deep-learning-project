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
epochs = 200
learning_rate = 0.1
size = 8

#Load Data
starting_image = Data.getSpecificImage(563, image_size = img_size)
ending_image = Data.getSpecificImage(1244, image_size = img_size)
#563,795

#Setup Base Model
model = M.AutoEncoder()
model.to(dev)
opt = optim.Adam(model.parameters(), lr = learning_rate)
#U.load_model_state(model, opt, 'trained_models/VeryDenoisingAE.pt')

#Setup DeepDream Model
DDModel = M.DeepDream(starting_image, ending_image, model)
DDOpt = optim.Adam(DDModel.parameters(), lr = learning_rate)

#Train Model
interpolated = U.interpolationMerge(starting_image, ending_image, model, size=size)
realinterpolated = U.getInterpolations(starting_image, ending_image, size=size)
history = U.fit_DeepDream(DDModel, F.mse_loss, DDOpt, epochs, history_size=size)
historyAE = model(history)
#Vis.displayImages(history, columMax=10)
#Vis.displayImages(historyAE, columMax=10)
Vis.displayImages(torch.cat((history,historyAE,interpolated,realinterpolated),0),columMax=size)
#Vis.displayImages(torch.cat((history,realinterpolated),0),columMax=size)

#To do:
# - Clean models file
# - Variational autoencoder
# - Choose architecture for standard ae
# - Better saving system (Include image_size & Loss)
