import os
import torch
import random
from torchvision.io import read_image
from torchvision import transforms as T
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
import utility as U


class AnimalImageDataset(Dataset):
    def __init__(self, img_dir, image_size = None, data_size = None, noise = 0):
        self.img_dir = img_dir
        self.image_size = image_size
        self.data_size = data_size
        self.noise = noise
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        totalImages = 0
        for _, _, files in os.walk(img_dir):

            for name in files:
                # Mac has DS_Store hidden cache file
                if "animal" in name:
                   totalImages += 1
        
        self.noOfImages = totalImages

    def __len__(self):

        if self.data_size and self.data_size <= self.noOfImages:
            return self.data_size
        return self.noOfImages

    def __getitem__(self, index):
        img_index = "(" + str(index + 1) + ")"
        img_path = os.path.join(self.img_dir, "animal " + img_index + ".jpg")
        target_image = read_image(img_path)

        #Preprocessing
        if self.image_size:
            target_image = T.Resize(self.image_size)(target_image)
        target_image = (lambda x: x/255)(target_image)

        #Adjusting Input image
        if self.noise:
            image = target_image + (torch.rand(target_image.size()) * (self.noise)) - (self.noise/2)
        else:
            image = target_image
        
        image = image.to(self.dev)
        target_image = target_image.to(self.dev)

        return image, target_image

class AnimalImageLabelDataset(Dataset):
    def __init__(self, img_dir, labels_file, image_size = None, data_size = None, noise = 0):
        self.img_dir = img_dir
        self.image_size = image_size
        self.image_labels = pd.read_csv(labels_file)
        self.data_size = data_size
        self.noise = noise
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.noOfImages = self.image_labels.shape[0]

    def __len__(self):

        if self.data_size and self.data_size <= self.noOfImages:
            return self.data_size
        return self.noOfImages

    def __getitem__(self, index):
        img_index = "(" + str(index + 1) + ")"
        img_path = os.path.join(self.img_dir, "animal " + img_index + ".jpg")
        image = read_image(img_path)
        label = torch.tensor(U.labelMapping(self.image_labels.iloc[index, 1]))
        #label = F.one_hot(label, num_classes=8)

        #Preprocessing
        if self.image_size:
            image = T.Resize(self.image_size)(image)
        image = (lambda x: x/255)(image)

        #Adjusting Input image
        if self.noise:
            image = image + (torch.rand(image.size()) * (self.noise)) - (self.noise/2)
        
        image = image.to(self.dev)
        label = label.to(self.dev)

        return image, label

def preprocessImage(image):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    image = (lambda x: x/255)(image)
    image = image.to(dev)
    return image

def getSpecificImage(index, image_size):
    img_index = "(" + str(index) + ")"
    img_path = os.path.join("data/val", "animal " + img_index + ".jpg")
    image = read_image(img_path)
    image = preprocessImage(image)
    if image_size:
        image = T.Resize(image_size)(image)
    return image.unsqueeze(0)


def getCat(image_size = None):
    return getSpecificImage(61, image_size)

def getDog(image_size = None):
    return getSpecificImage(96, image_size)

def getFox(image_size = None):
    return getSpecificImage(219, image_size)

def getWolf(image_size = None):
    return getSpecificImage(555, image_size)

def getLeopard(image_size = None):
    return getSpecificImage(590, image_size)

def getTiger(image_size = None):
    return getSpecificImage(491, image_size)

def getLion(image_size = None):
    return getSpecificImage(565, image_size)

def getRandom(image_size = None):
    return getSpecificImage(random.randint(1, 1500), image_size)

def getNoise(image_size = None):
    noise = torch.rand((1, 3) + image_size)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    noise = noise.to(dev)
    return noise