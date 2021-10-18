import os
import torch
import random
from torchvision.io import read_image
from torchvision import transforms as T
from torch.utils.data import Dataset


class AnimalImageDataset(Dataset):
    def __init__(self, img_dir, image_size = None, data_size = None):
        self.img_dir = img_dir
        self.image_size = image_size
        self.data_size = data_size

        totalImages = 0
        for _, _, files in os.walk(img_dir):

            for _ in files:
                totalImages += 1
        
        self.noOfImages = totalImages

    def __len__(self):

        if self.data_size and self.data_size <= self.noOfImages:
            return self.data_size
        return self.noOfImages

    def __getitem__(self, index):
        img_index = "(" + str(index + 1) + ")"
        img_path = os.path.join(self.img_dir, "animal " + img_index + ".jpg")
        image = read_image(img_path)

        if self.image_size:
            image = T.Resize(self.image_size)(image)
        
        image = preprocessImage(image)

        return image

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