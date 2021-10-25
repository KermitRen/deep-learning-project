from matplotlib.pyplot import sca
from torch import nn
import torch
import torch.nn.functional as F
import torchvision.models as models

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

class MobileAutoEncoder(nn.Module):
    def __init__(self, dev):
        super().__init__()
        self.model = models.mobilenet.mobilenet_v2(pretrained=True)
        self.model.classifier = nn.Sequential(
                                nn.Conv2d(1280, 640, kernel_size=3, stride=1, padding="same"),
                                nn.BatchNorm2d(640),
                                nn.ReLU(),
                                nn.Upsample(scale_factor=2, mode="nearest"),
                                nn.Conv2d(640, 320, kernel_size=3, stride=1, padding="same"),
                                nn.BatchNorm2d(320),
                                nn.ReLU(),
                                nn.Upsample(scale_factor=2, mode="nearest"),
                                nn.Conv2d(320, 160, kernel_size=3, stride=1, padding="same"),
                                nn.BatchNorm2d(160),
                                nn.ReLU(),
                                nn.Upsample(scale_factor=2, mode="nearest"),
                                nn.Conv2d(160, 80, kernel_size=3, stride=1, padding="same"),
                                nn.BatchNorm2d(80),
                                nn.ReLU(),
                                nn.Upsample(scale_factor=2, mode="nearest"),
                                nn.Conv2d(80, 40, kernel_size=3, stride=1, padding="same"),
                                nn.BatchNorm2d(40),
                                nn.ReLU(),
                                nn.Upsample(scale_factor=2, mode="nearest"),
                                nn.Conv2d(40, 3, kernel_size=3, stride=1, padding="same"),
                                nn.BatchNorm2d(3),
                                nn.Sigmoid()
                                )
        self.model.to(dev)
        self.model.eval()
        for param in self.model.features.parameters():
            param.requires_grad = False

    def encode(self, x):
        return self.model.features(x)

    def decode(self, x):
        return self.model.classifier(x)

    def forward(self, x):
        return self.model.classifier(self.model.features(x))

class AlexNet(nn.Module):
    def __init__(self, dev):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        self.model.to(dev)
        for param in self.model.parameters():
            param.requires_grad = False

    def encode(self, x):
        return self.model.features(x)

    def forward(self, x):
        return self.model(x)

class MobileNet(nn.Module):
    def __init__(self, dev):
        super().__init__()
        self.model = models.mobilenet.mobilenet_v2(pretrained=True)
        self.model.to(dev)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def encode(self, x):
        return self.model.features(x)

    def forward(self, x):
        return self.model(x)

class DeepDream(nn.Module):
    def __init__(self, starting_image, ending_image, model):
        super().__init__()
        self.W = nn.Parameter(starting_image)
        self.model = model
        self.ending_encoding = self.model.encode(ending_image)
        print(self.ending_encoding.size())

    def forward(self):
        return self.model.encode(self.W)

    def getWeights(self):
        return self.W

    def getEndingEncoding(self):
        return self.ending_encoding