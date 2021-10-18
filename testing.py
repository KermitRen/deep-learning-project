import torch
from torch import optim
from torch.utils import data
import visualizing as Vis
import datasets as Data
from models import AutoEncoder, AutoEncoder2, AutoEncoder3, AutoEncoder4, AutoEncoder5
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utility as U

#Setup GPU
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

img_size = (256, 256)

#Load Data
animal_dataset_train = Data.AnimalImageDataset(img_dir = 'data/train', image_size = img_size, data_size=1000)
animal_dataset_val = Data.AnimalImageDataset(img_dir = 'data/val', image_size = img_size)
loader_train = DataLoader(dataset = animal_dataset_train, batch_size = 64, shuffle = True)
loader_val = DataLoader(dataset = animal_dataset_val, batch_size = 64, shuffle = False)

#Setup Model
model = AutoEncoder3()
model.to(dev)
opt = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)

#U.load_model_state(model, opt, 'trained_models/autoencoder6.pt')

history = U.fit_model(loader_train, model, F.mse_loss, opt, 5, loader_val)
U.save_model_state(model, opt, 'trained_models/autoencoder9.pt')
Vis.displayHistory(history)

catImage = Data.getRandom(image_size = img_size)
foxImage = Data.getRandom(image_size = img_size)

catEncoding = model.encode(catImage)
foxEncoding = model.encode(foxImage)
catDecoding = model.decode(catEncoding)
foxDecoding = model.decode(foxEncoding)

hybridEncoding1 = torch.lerp(catEncoding, foxEncoding, 0.25)
hybridEncoding2 = torch.lerp(catEncoding, foxEncoding, 0.5)
hybridEncoding3 = torch.lerp(catEncoding, foxEncoding, 0.75)

hybrid1 = model.decode(hybridEncoding1)
hybrid2 = model.decode(hybridEncoding2)
hybrid3 = model.decode(hybridEncoding3)

merging = torch.cat((catDecoding, hybrid1, hybrid2, hybrid3, foxDecoding),0)
Vis.displayImages(merging, 1)


#To do:
# - Clean models file
# - Create deepdream model
# - Clean up test file
# - Variational autoencoder
# - Choose architecture for standard ae
# - Better saving system (Include image_size & Loss)
