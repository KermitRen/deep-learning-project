from torch import nn
import torch
import torch.nn.functional as F

#Original AutoEncoder3
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding="same")
        self.conv2 = nn.Conv2d(16, 2, kernel_size=3, stride=1, padding="same")
        self.conv3 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding="same")
        self.conv4 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding="same")
        self.conv5 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding="same")

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(2)
        self.bn3 = nn.BatchNorm2d(2)
        self.bn4 = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(3)

    def forward(self, x):
        return self.decode(self.encode(x))
    
    def encode(self, x):
        #Encoder
        layer1 = F.relu(self.bn1(self.conv1(x)))
        layer1 = nn.MaxPool2d(2)(layer1)
        layer2 = F.relu(self.bn2(self.conv2(layer1)))
        layer2 = nn.MaxPool2d(2)(layer2)

        return layer2
    
    def decode(self, x):
        #Decoder
        layer3 = F.relu(self.bn3(self.conv3(x)))
        layer3 = nn.Upsample(scale_factor = 2, mode = 'nearest')(layer3)
        layer4 = F.relu(self.bn4(self.conv4(layer3)))
        layer4 = nn.Upsample(scale_factor = 2, mode = 'nearest')(layer4)

        layer5 = torch.sigmoid(self.bn5(self.conv5(layer4)))
        return layer5

class DeepDream(nn.Module):
    def __init__(self, starting_image, ending_image, model):
        super().__init__()
        self.W = nn.Parameter(starting_image)
        self.model = model
        self.ending_encoding = self.model.encode(ending_image)

    def forward(self):

        return self.model.encode(self.W)

    def getWeights(self):
        return self.W

    def getEndingEncoding(self):
        return self.ending_encoding