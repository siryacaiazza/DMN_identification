import torch
import torch.nn as nn


class CNN3D(nn.Module):
    """
    3D CNN for volumetric image classification.
    Input shape: (batch, channels, 80, 80, 56)

    """

    def __init__(self, num_classes: int = 2, dropout_p: float = 0.5):
        super().__init__()

        # --- Convolutional layers ---

        # Layer 1: 20 kernels, 3x3x3, followed by MaxPool(2)
        self.conv1 = nn.Sequential(
            nn.Conv3d(2, 20, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),   # 80x80x56 -> 40x40x28
        )

        # Layer 2: 20 kernels, 3x3x3
        self.conv2 = nn.Sequential(
            nn.Conv3d(20, 20, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Dropout between conv2 and conv3
        self.dropout = nn.Dropout3d(p=dropout_p)

        # Layer 3: 10 kernels, 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(20, 10, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # --- Fully connected layers ---

        fc_input_size = 10 * 40 * 40 * 28  # = 324_000

        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward_features(self, x):
        """
        Restituisce l'embedding a 256-d, prima del classificatore finale.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = x.flatten(1)
        x = self.fc[0](x)  # Linear(fc_input_size, 256)
        x = self.fc[1](x)  # ReLU
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc[2](x)  # Linear(256, num_classes)
        return x


if __name__ == "__main__":
    model = CNN3D(num_classes=2)
    print(model)
    print()


    # Forward pass test
    dummy = torch.randn(16, 2, 80, 80, 56)   # batch=16, 4 channels
    out = model(dummy)
    print(f"Input shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")