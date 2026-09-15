import torch
import torch.nn as nn

from config import GENRES


class GenreCNN(nn.Module):
    """
    VGG-style CNN: blocks 1-3 use TWO conv layers before pooling (more
    representational depth at each resolution before downsampling), block 4
    uses one. Channel depth increases 32 -> 64 -> 128 -> 256 -- roughly 6x
    the capacity of the original 106k-parameter version. Dropout is lighter
    in early blocks (where too much regularization causes underfitting) and
    concentrated mainly in the classifier head.
    """

    def __init__(self, n_classes=len(GENRES)):
        super().__init__()

        def conv_bn_relu(in_c, out_c):
            return [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ]

        self.features = nn.Sequential(
            *conv_bn_relu(1, 32),
            *conv_bn_relu(32, 32),
            nn.MaxPool2d(2),
            nn.Dropout(0.05),

            *conv_bn_relu(32, 64),
            *conv_bn_relu(64, 64),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),

            *conv_bn_relu(64, 128),
            *conv_bn_relu(128, 128),
            nn.MaxPool2d(2),
            nn.Dropout(0.15),

            *conv_bn_relu(128, 256),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),   # was 0.4
            nn.Linear(128, n_classes),
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