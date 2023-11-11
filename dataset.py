from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        # Collect all image file paths in the real images directory
        self.real_images = [os.path.join(real_dir, file) for file in os.listdir(real_dir)]
        # Collect all image file paths in the fake images directory
        self.fake_images = [os.path.join(fake_dir, file) for file in os.listdir(fake_dir)]
        # Combine lists of real and fake image paths
        self.total_images = self.real_images + self.fake_images
        # Create labels for the images (0 for real, 1 for fake)
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        # Assign the transformation to be applied on images
        self.transform = transform

    def __len__(self):
        # Return the total number of images in the dataset
        return len(self.total_images)

    def __getitem__(self, idx):
        # Retrieve the path of the image at the given index
        image_path = self.total_images[idx]
        # Retrieve the corresponding label of the image
        label = self.labels[idx]
        # Load the image from the file and convert it to RGB format
        image = Image.open(image_path).convert('RGB')
        # Apply the transformation to the image if any
        if self.transform:
            image = self.transform(image)
        # Return the transformed image and its label
        return image, label
