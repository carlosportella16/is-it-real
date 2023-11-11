import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Initialize convolutional layer 1 with 16 filters, kernel size 3x3, and padding of 1
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # Initialize a max pooling layer with 2x2 window and stride 2
        self.pool = nn.MaxPool2d(2, 2)
        # Initialize convolutional layer 2 with 32 filters, kernel size 3x3, and padding of 1
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # Initialize convolutional layer 3 with 64 filters, kernel size 3x3, and padding of 1
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # Initialize the first fully connected layer (dense layer)
        self.fc1 = nn.Linear(64 * 4 * 4, 120)
        # Initialize the second fully connected layer
        self.fc2 = nn.Linear(120, 84)
        # Initialize the third fully connected layer with a single output (binary classification)
        self.fc3 = nn.Linear(84, 1)
        # Initialize a dropout layer with 50% probability of an element to be zeroed
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Define the forward pass
        # Apply the first convolutional layer followed by ReLU activation function and pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply the second convolutional layer followed by ReLU and pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Apply the third convolutional layer followed by ReLU and pooling
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten the output for the dense layers
        x = x.view(-1, 64 * 4 * 4)
        # Apply the first fully connected layer with ReLU
        x = F.relu(self.fc1(x))
        # Apply dropout to the output of the first dense layer
        x = self.dropout(x)
        # Apply the second fully connected layer with ReLU
        x = F.relu(self.fc2(x))
        # Apply dropout to the output of the second dense layer
        x = self.dropout(x)
        # Apply the third fully connected layer and sigmoid activation for binary classification
        x = torch.sigmoid(self.fc3(x))
        return x
