import torch
import torch.nn as nn

from config import GENRES


class GenreCNN(nn.Module):
    """
    4 convolutional blocks (conv -> batchnorm -> ReLU -> maxpool),
    followed by global average pooling and a fully connected classifier.

    Each conv block doubles the channel depth while pooling halves the
    spatial dimensions, so the network progressively trades spatial detail
    for more abstract, higher-level feature channels -- matching the
    proposal's description of learning beat structure, instrument texture,
    and tonal quality through successive layers.
    """

    def __init__(self, n_classes=len(GENRES)):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1 -> 16 channels
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),

            # Block 2: 16 -> 32 channels
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),

            # Block 3: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            # Block 4: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # collapses spatial dims to 1x1, regardless of input size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
            # no softmax here -- nn.CrossEntropyLoss applies it internally during training
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = GenreCNN()
    print(model)
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")