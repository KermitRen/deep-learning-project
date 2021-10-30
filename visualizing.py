import torchvision
import matplotlib.pyplot as plt
import torch

def displayImages(images, columMax = 10):
    images = images.cpu()

    size = len(images)
    rows = int(size/columMax) + 1
    columns = min(size, columMax)
    grid_img = torchvision.utils.make_grid(images[0:columns*rows], nrow=columns)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()
