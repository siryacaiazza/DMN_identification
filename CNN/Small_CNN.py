from dataloader import FMRIDataset, DataLoader3D
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Simple_CNN(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5):
        super(Simple_CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(89600, 256)  
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)
        self.flatten = nn.Flatten()

    def forward_features(self, x):
        """Restituisce l'embedding a 256-d, prima del classificatore finale."""
        out = self.layer1(x)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.dropout(out)
        out = self.layer3(out)
        out = self.dropout(out)
        out = self.flatten(out)
        out = F.relu(self.fc1(out))  # embedding a 256, con la stessa ReLU del forward originale
        return out

    def forward(self, x):
        out = self.forward_features(x)
        out = self.fc2(out)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Do the same for this model as for the pretrained one
simple_model = Simple_CNN(num_classes=2, dropout_rate=0.5).to(device)

inp = torch.randn(16, 2, 80, 80, 56).to(device)
out = simple_model(inp)
print(out.shape)

print('Input shape is', inp.shape)
print('Output shape is', out.shape)

total_params = sum(p.numel() for p in simple_model.parameters())
print(f"Total number of parameters in the model: {total_params}")