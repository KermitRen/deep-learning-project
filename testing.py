from matplotlib.pyplot import hist
import torch
from torch import optim
from torch.utils import data
import visualizing as Vis
import datasets as Data
from models import DeepDream, MobileNet, MobileAutoEncoder
from torch.utils.data import DataLoader, dataset
import torch.nn.functional as F
import utility as U
import torchvision.models as models
import torch.utils.model_zoo

#Setup GPU
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

img_size = (256, 256)

#Load Data
animal_dataset_train = Data.AnimalImageDataset(img_dir = 'data/train', image_size = img_size)
animal_dataset_val = Data.AnimalImageDataset(img_dir = 'data/val', image_size = img_size)
loader_train = DataLoader(dataset = animal_dataset_train, batch_size = 64, shuffle = True)
loader_val = DataLoader(dataset = animal_dataset_val, batch_size = 64, shuffle = False)

#Setup Model
model = MobileAutoEncoder(dev)
model.to(dev)
opt = optim.SGD(model.parameters(), lr = 0.1, momentum=0.9)

print(model.encode(next(iter(loader_train))).size())
#U.load_model_state(model, opt, 'trained_models/autoencoder.pt')
#for param in model.parameters():
#    param.requires_grad = False
history = U.fit_AutoEncoder(loader_train, model, F.mse_loss, opt, 3, loader_val)
U.save_model_state(model, opt, 'trained_models/Mautoencoder.pt')
Vis.displayHistory(history)

#Setup Model
#model2 = MobileNet(dev)
#model2.to(dev)
#print(model2.encode(Data.getCat(image_size=img_size)))

#Setup DeepDream Model
starting_image = Data.getCat(image_size = img_size)
ending_image = Data.getDog(image_size = img_size)
DDModel = DeepDream(starting_image, ending_image, model)

ddopt = optim.SGD(DDModel.parameters(), lr = 0.1, momentum=0.9)
#print("DeepDream Weights", DDModel.getWeights())
#weights = DDModel.getWeights()
#weights.requires_grad = False
#Vis.displayImages(weights, 1)
#weights.requires_grad = True
history = U.fit_DeepDream(DDModel, F.mse_loss, ddopt, 19000) #No batch, more epochs
Vis.displayHistory(history)
newImages = []
for i in range(len(history)):
    newImages.append(model(history[i]))
Vis.displayHistory(newImages)


#history = U.fit_model(loader_train, model, F.mse_loss, opt, 5, loader_val)
#U.save_model_state(model, opt, 'trained_models/deepdream.pt')
#Vis.displayHistory(history)

# catImage = Data.getRandom(image_size = img_size)
# foxImage = Data.getRandom(image_size = img_size)

# catEncoding = model.encode(catImage)
# foxEncoding = model.encode(foxImage)
# catDecoding = model.decode(catEncoding)
# foxDecoding = model.decode(foxEncoding)

# hybridEncoding1 = torch.lerp(catEncoding, foxEncoding, 0.25)
# hybridEncoding2 = torch.lerp(catEncoding, foxEncoding, 0.5)
# hybridEncoding3 = torch.lerp(catEncoding, foxEncoding, 0.75)

# hybrid1 = model.decode(hybridEncoding1)
# hybrid2 = model.decode(hybridEncoding2)
# hybrid3 = model.decode(hybridEncoding3)

# merging = torch.cat((catDecoding, hybrid1, hybrid2, hybrid3, foxDecoding),0)
# Vis.displayImages(merging, 1)


#To do:
# - Clean models file
# - Create deepdream model
# - Clean up test file
# - Variational autoencoder
# - Choose architecture for standard ae
# - Better saving system (Include image_size & Loss)
