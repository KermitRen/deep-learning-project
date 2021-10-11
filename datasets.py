import os
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset


class AnimalImageDataset(Dataset):
    def __init__(self, img_dir, size = None):
        self.img_dir = img_dir
        self.size = size

        totalImages = 0
        for _, _, files in os.walk(img_dir):

            for _ in files:
                totalImages += 1
        
        self.noOfImages = totalImages

    def __len__(self):
        return self.noOfImages

    def __getitem__(self, index):
        img_index = "(" + str(index + 1) + ")"
        img_path = os.path.join(self.img_dir, "animal " + img_index + ".jpg")
        image = read_image(img_path)

        if self.size:
            image = transforms.Compose([transforms.Resize(self.size)])(image)

        return image

