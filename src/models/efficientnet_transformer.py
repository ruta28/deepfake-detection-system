import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNet_Transformer(nn.Module):
    def __init__(self, num_classes=1, effnet_output_size=1280, num_frames=16, 
                 d_model=256, nhead=8, num_layers=3, dropout=0.1):
        super(EfficientNet_Transformer, self).__init__()
        
        # 1. Load pre-trained EfficientNet-B0
        weights = EfficientNet_B0_Weights.DEFAULT
        effnet = efficientnet_b0(weights=weights)
        self.features = nn.Sequential(*list(effnet.children())[:-1])
        
        # Freeze EfficientNet
        for param in self.features.parameters():
            param.requires_grad = False
            
        # 2. Linear projection layer
        # Project EfficientNet's 1280 features to d_model (256)
        self.projection = nn.Linear(effnet_output_size, d_model)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True # This is important!
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 4. Positional Encoding
        # Learnable positional encoding for the 16 frames
        self.pos_encoder = nn.Parameter(torch.zeros(1, num_frames, d_model))
        
        # 5. Classification Head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
        
        self.d_model = d_model

    def forward(self, x):
        # x shape: (batch_size, num_frames, C, H, W)
        b, t, c, h, w = x.shape
        
        # Reshape for EfficientNet
        x = x.view(b * t, c, h, w)
        
        # Get features
        x = self.features(x)
        
        # Reshape for Transformer
        # (batch_size * num_frames, features) -> (batch_size, num_frames, features)
        x = x.view(b, t, -1)
        
        # Project features to d_model
        x = self.projection(x) # (b, t, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoder
        
        # Pass through Transformer
        x = self.transformer_encoder(x)
        
        # Take the output of the *first* token (like BERT) or mean
        # Let's use the mean of all frame outputs
        x = x.mean(dim=1) # (b, d_model)
        
        # Pass through final classification layer
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
