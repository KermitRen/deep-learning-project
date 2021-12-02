import torch
from torch import optim
from torch import random
import torch.nn.functional as F
import visualizing as Vis
import datasets as Data
import models as M
import utility as U

#Setup Variables
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
img_size = (256, 256)
epochs = 150
learning_rate = 0.01
size = 7

#Load Data
starting_image = Data.getSpecificImage(360,image_size = img_size)
ending_image = Data.getSpecificImage(592,image_size = img_size)
#1088, 592
#1452, 592
#Tiger, Lion

#
#Decoder Images
# mod = M.AutoEncoder()
# mod.to(dev)
# o = optim.Adam(mod.parameters(), lr = learning_rate)
# U.load_model_state(mod, o, 'trained_models/AE.pt')

# begins = []
# ends = []
# for i in range(5):
#     random_image = Data.getRandom(image_size=img_size)
#     dec = mod(random_image)
#     begins.append(random_image)
#     ends.append(dec)

# Vis.displayImages(torch.cat(tuple(begins + ends),0),columMax=5)

#Setup Base Models
#model = M.VGGClassifier(dev)
#model.to(dev)

model1 = M.AutoEncoder()
model1.to(dev)
opt1 = optim.Adam(model1.parameters(), lr = learning_rate)
U.load_model_state(model1, opt1, 'trained_models/AE.pt')

model2 = M.AutoEncoder()
model2.to(dev)
opt2 = optim.Adam(model2.parameters(), lr = learning_rate)
U.load_model_state(model2, opt2, 'trained_models/DenoisingAE.pt')

model3 = M.AutoEncoder()
model3.to(dev)
opt3 = optim.Adam(model3.parameters(), lr = learning_rate)
U.load_model_state(model3, opt3, 'trained_models/VeryDenoisingAE.pt')

#Setup DeepDream Model
models = [model1, model2, model3]
results = (U.getInterpolations(starting_image, ending_image, size=size),)
results = results + (U.interpolationMerge(starting_image, ending_image, model1, size=size),)

for i in range(len(models)):
    DDModel = M.DeepDream(starting_image.clone(), ending_image.clone(), models[i])
    DDOpt = optim.Adam(DDModel.parameters(), lr = learning_rate)
    history = U.fit_DeepDream(DDModel, F.mse_loss, DDOpt, epochs, history_size=size)
    results = results + (history,)

    historyEncoded = models[i](history)
    results = results + (historyEncoded,)

#Train Model
#interpolated = U.interpolationMerge(starting_image, ending_image, model, size=size)
#historyAE = model(history)
#Vis.displayImages(history, columMax=10)
#Vis.displayImages(historyAE, columMax=10)
Vis.displayImages(torch.cat(results,0),columMax=size)
#Vis.displayImages(torch.cat((history,realinterpolated),0),columMax=size)

#To do:
# - Clean models file
# - Variational autoencoder
# - Choose architecture for standard ae
# - Better saving system (Include image_size & Loss)
