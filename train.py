# Import necessary modules
from dataset import CustomDataset
from model import Net
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

def train_and_save(real_dir, fake_dir, model, model_file, epochs=10, batch_size=4, learning_rate=0.001):
    # Define the image transformations: resizing, converting to tensor, and normalizing
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize the image to 32x32 pixels
        transforms.ToTensor(),        # Convert the image to a PyTorch tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the tensor
    ])

    # Create a custom dataset using the provided directories and transformations
    dataset = CustomDataset(real_dir, fake_dir, transform=transform)
    # Create a DataLoader to batch and shuffle the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define the loss criterion and optimizer for training
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  # Stochastic Gradient Descent optimizer

    # Check if CUDA (GPU support) is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move the model to the specified device (GPU or CPU)
    model.to(device)

    # Calculate the total number of training steps
    total_steps = len(dataloader) * epochs
    current_step = 0

    # Begin the training loop
    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            # Move the input and label data to the specified device
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1, 1).type(torch.FloatTensor).to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: compute the output of the model
            outputs = model(inputs)
            # Compute the loss
            loss = criterion(outputs, labels)
            # Backward pass: compute the gradients
            loss.backward()
            # Update the model parameters
            optimizer.step()

            # Update the current step and calculate the training progress
            current_step += 1
            progress = (current_step / total_steps) * 100
            # Print the progress and loss
            print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], Progress: {progress:.2f}%, Loss: {loss.item():.4f}')

    # Print that the training is finished
    print('Finished Training')
    # Save the trained model state to a file
    torch.save(model.state_dict(), model_file)
    print(f'Model saved to {model_file}')

# If the script is run directly, execute the training function
if __name__ == '__main__':
    # Initialize the model
    model = Net()
    # Call the training function with specified parameters
    train_and_save('./data/real/', './data/fake/', model, 'model.pth', epochs=15, batch_size=32, learning_rate=0.001)
