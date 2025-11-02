import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

"""
This is a STATIC IMAGE model.
It uses EfficientNet but has NO LSTM.
It is designed to be trained on and predict single frames.
"""
class StaticImageModel(nn.Module):
    def __init__(self, num_classes=1):
        super(StaticImageModel, self).__init__()
        
        # 1. Load pre-trained EfficientNet-B0
        weights = EfficientNet_B0_Weights.DEFAULT
        base = efficientnet_b0(weights=weights)
        
        # 2. We will use its features, up to the final block
        self.features = nn.Sequential(*list(base.children())[:-1])
        
        # Get the output feature size (it's 1280 for B0)
        num_features = base.classifier[1].in_features
        
        # 3. Create a new classifier head for static images
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Pool features
            nn.Flatten(),
            nn.Dropout(0.5), # Add dropout to prevent overfitting
            nn.Linear(num_features, num_classes) # Final prediction
        )

    def forward(self, x):
        # x shape: (batch_size, C, H, W) - a 4D tensor
        x = self.features(x)
        x = self.classifier(x)
        return x

