# Import necessary modules
from model import Net
from torchvision import transforms
from PIL import Image
import torch

def predict_image(image_path, model):
    # Define the transformations to be applied to the input image
    transform = transforms.Compose([
        # Resize the image to 32x32 pixels
        transforms.Resize((32, 32)),
        # Convert the image to a PyTorch Tensor
        transforms.ToTensor(),
        # Normalize the tensor with mean and standard deviation for each color channel
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Open the image from the provided path and convert it to RGB format
    image = Image.open(image_path).convert('RGB')
    # Apply the transformations to the image
    image = transform(image)
    # Add a batch dimension to the image tensor (required for model input)
    image = image.unsqueeze(0)

    # Pass the transformed image through the model to get the output
    output = model(image)
    # Apply the sigmoid function to the output to get the probability
    probability = torch.sigmoid(output).item()
    # Return the probability
    return probability
