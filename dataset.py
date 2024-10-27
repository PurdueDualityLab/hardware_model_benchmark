import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.models import VGG16_Weights, vgg16
from tqdm import tqdm


class ImageNetValidation(Dataset):
    def __init__(self, image_dir, ground_truth_file, transform=None, num_samples=1000):
        self.image_dir = image_dir
        self.transform = transform
        self.num_samples = num_samples

        # Load ground truth labels
        with open(ground_truth_file, 'r') as f:
             self.labels = [int(line.strip()) for line in f][:num_samples]

        # Get image files
        self.image_files = sorted([f for f in os.listdir(image_dir)
                                 if f.endswith('.JPEG')])[:num_samples]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]

