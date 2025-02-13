# Define the model architecture (Ensure it matches what was used for training)
def load_model():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    class AdvancedCNN(nn.Module):
        def __init__(self, dropout_rate=0.2):
            super(AdvancedCNN, self).__init__()

            # First Conv Block
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool1 = nn.MaxPool2d(2, 2)

            # Second Conv Block (Added Extra Layer)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.conv3_extra = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # Extra Layer
            self.bn3_extra = nn.BatchNorm2d(128)
            self.pool2 = nn.MaxPool2d(2, 2)

            # Third Conv Block (Added Extra Layer)
            self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bn5 = nn.BatchNorm2d(256)
            self.conv5_extra = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # Extra Layer
            self.bn5_extra = nn.BatchNorm2d(256)
            self.pool3 = nn.MaxPool2d(2, 2)

            # Fully Connected Layers
            self.fc1 = nn.Linear(256 * 4 * 4, 512)
            self.bn_fc1 = nn.BatchNorm1d(512)
            self.dropout = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(512, 10)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool1(F.relu(self.bn2(self.conv2(x))))

            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn3_extra(self.conv3_extra(x)))  # Extra Layer
            x = self.pool2(x)

            x = F.relu(self.bn5(self.conv5(x)))
            x = F.relu(self.bn5_extra(self.conv5_extra(x)))  # Extra Layer
            x = self.pool3(x)

            x = x.view(-1, 256 * 4 * 4)
            x = F.relu(self.bn_fc1(self.fc1(x)))
            x = self.dropout(x)
            x = self.fc2(x)

            return x

    model = AdvancedCNN()  # Instantiate the model
    model.load_state_dict(torch.load("CIFAR-10_model.pth", map_location=torch.device('cpu')))  # Load weights
    model.eval()
    return model

# Instantiate model for direct testing (optional)
if __name__ == "__main__":
    model = load_model()
    print("Model loaded successfully!")
