from torch import nn
import torch
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding="same")
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding="same")
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding="same")
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding="same")
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding="same")
        self.conv6 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding="same")

    def forward(self, x):

        #Encoder
        layer1 = F.relu(self.conv1(x))
        layer1 = nn.MaxPool2d(2)(layer1)
        layer2 = F.relu(self.conv2(layer1))
        layer2 = nn.MaxPool2d(2)(layer2)
        layer3 = F.relu(self.conv3(layer2))
        layer3 = nn.MaxPool2d(2)(layer3)

        #Decoder
        layer4 = F.relu(self.conv4(layer3))
        layer4 = nn.Upsample(scale_factor = 2, mode = 'nearest')(layer4)
        layer5 = F.relu(self.conv5(layer4))
        layer5 = nn.Upsample(scale_factor = 2, mode = 'nearest')(layer5)
        layer6 = F.relu(self.conv6(layer5))
        layer6 = nn.Upsample(scale_factor = 2, mode = 'nearest')(layer6)
        return layer6



class AutoEncoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding="same")
        self.conv2 = nn.Conv2d(16, 2, kernel_size=3, stride=1, padding="same")
        self.conv3 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding="same")
        self.conv4 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding="same")
        self.conv5 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding="same")

    def forward(self, x):
        return self.decode(self.encode(x))
    
    def encode(self, x):
        #Encoder
        layer1 = F.relu(self.conv1(x))
        layer1 = nn.MaxPool2d(2)(layer1)
        layer2 = F.relu(self.conv2(layer1))
        layer2 = nn.MaxPool2d(2)(layer2)

        return layer2
    
    def decode(self, x):
        #Decoder
        layer3 = F.relu(self.conv3(x))
        layer3 = nn.Upsample(scale_factor = 2, mode = 'nearest')(layer3)
        layer4 = F.relu(self.conv4(layer3))
        layer4 = nn.Upsample(scale_factor = 2, mode = 'nearest')(layer4)

        layer5 = torch.sigmoid(self.conv5(layer4))
        return layer5

class AutoEncoder3(nn.Module):
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

class AutoEncoder4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding="same")
        self.conv2 = nn.Conv2d(16, 4, kernel_size=5, stride=1, padding="same")
        self.conv3 = nn.Conv2d(4, 4, kernel_size=5, stride=1, padding="same")
        self.conv4 = nn.Conv2d(4, 16, kernel_size=5, stride=1, padding="same")
        self.conv5 = nn.Conv2d(16, 3, kernel_size=5, stride=1, padding="same")

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm2d(4)
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



class AutoEncoder5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding="same")
        self.conv2 = nn.Conv2d(16, 4, kernel_size=3, stride=1, padding="same")
        self.conv3 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding="same")
        self.conv4 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding="same")
        self.conv5 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding="same")

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm2d(4)
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