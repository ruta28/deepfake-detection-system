import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

"""
This file contains the two separate model architectures
1. StaticImageModel: For single image predictions.
2. EfficientNet_LSTM: For video predictions.
"""

# --- 1. The new STATIC IMAGE Model ---
# This is a complete model for static images, based on your FrameCNN
class StaticImageModel(nn.Module):
    def __init__(self, num_classes=1):
        super(StaticImageModel, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        weights = EfficientNet_B0_Weights.DEFAULT
        base = efficientnet_b0(weights=weights)
        
        # We will use its features, up to the final block
        self.features = nn.Sequential(*list(base.children())[:-1])
        
        # Get the output feature size
        num_features = base.classifier[1].in_features
        
        # Create a new classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Pool features
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(num_features, 1) # Final prediction
        )

    def forward(self, x):
        # x shape: (batch_size, C, H, W)
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 2. Your existing VIDEO Model ---
class EfficientNet_LSTM(nn.Module):
    def __init__(self, lstm_hidden_size=256, lstm_layers=2, bidirectional=True):
        super(EfficientNet_LSTM, self).__init__()

        weights = EfficientNet_B0_Weights.DEFAULT
        self.efficientnet = efficientnet_b0(weights=weights)

        # Freeze the backbone
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
        # x shape: (batch_size, seq_len, C, H, W)
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.efficientnet(x)
        features = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features)
        lstm_out = lstm_out[:, -1, :] # Get last time step
        output = self.classifier(lstm_out)
        return output
