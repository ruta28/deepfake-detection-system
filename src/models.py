import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

"""
This file contains BOTH model architectures for your app.
"""

# --- 1. YOUR VIDEO MODEL ---
class EfficientNet_LSTM(nn.Module):
    def __init__(self, lstm_hidden_size=256, lstm_layers=2, bidirectional=True):
        super(EfficientNet_LSTM, self).__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.efficientnet = efficientnet_b0(weights=weights)
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        lstm_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.efficientnet(x)
        features = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features)
        lstm_out = lstm_out[:, -1, :]
        output = self.classifier(lstm_out)
        return output

# --- 2. YOUR NEW STATIC IMAGE MODEL ---
class StaticImageModel(nn.Module):
    def __init__(self, num_classes=1):
        super(StaticImageModel, self).__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        base = efficientnet_b0(weights=weights)
        self.features = nn.Sequential(*list(base.children())[:-1])
        num_features = base.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        # x shape: (B, C, H, W)
        x = self.features(x)
        x = self.classifier(x)
        return x

