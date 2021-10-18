import torchvision
import matplotlib.pyplot as plt
import torch

def displayImages(images, rows):
    images = images.cpu()
    grid_img = torchvision.utils.make_grid(images[0:5*rows], nrow=5)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()

def compareImages(beforeImages, afterImages):

    #Covert single images to batches of 1
    if(len(beforeImages.size()) == 3):
        beforeImages = beforeImages.unsqueeze(0)

    if(len(afterImages.size()) == 3):
        afterImages = afterImages.unsqueeze(0)

    #Merge batches
    minNoOfImages = min(beforeImages.size()[0], afterImages.size()[0], 5)
    mergedImages = torch.cat((beforeImages[0:minNoOfImages], afterImages[0:minNoOfImages]), 0)
    displayImages(mergedImages, 2)

def displayHistory(history):

    displayImages(torch.cat(tuple(history), 0), len(history))