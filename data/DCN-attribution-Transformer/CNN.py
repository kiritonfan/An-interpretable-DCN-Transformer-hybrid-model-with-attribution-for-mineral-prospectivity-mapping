"""Define the standard convolutional baseline classifier."""


import torch.nn as nn


class CNN(nn.Module):
    """Three-layer CNN used for the controlled CNN-vs-DCN ablation."""

    def __init__(self, num_classes=2, output_dim=512):
        super().__init__()
        self.output_dim = output_dim
        self.block1 = nn.Sequential(
            nn.Conv2d(42, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.block3 = nn.Conv2d(128, output_dim, kernel_size=1)
        hidden_dim = max(64, output_dim // 2)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x, extract_features=False):
        x = self.block1(x)
        x = self.block2(x)
        features = self.block3(x)
        if extract_features:
            return features
        return self.classifier(features)


def create_cnn_model(num_classes=2, feature_dim=32):
    return CNN(num_classes=num_classes, output_dim=feature_dim)


if __name__ == "__main__":
    from Dense_transformers import run_model_pipeline

    run_model_pipeline(
        create_cnn_model,
        "cnn",
        pipeline_defaults={"epochs": 100, "patience": 20},
    )
