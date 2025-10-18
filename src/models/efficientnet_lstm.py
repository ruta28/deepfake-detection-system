import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNet_LSTM(nn.Module):
    def __init__(self, lstm_hidden_size=256, lstm_layers=2, bidirectional=True):
        super(EfficientNet_LSTM, self).__init__()
        
        # 1. Load pre-trained EfficientNet-B0
        weights = EfficientNet_B0_Weights.DEFAULT
        self.efficientnet = efficientnet_b0(weights=weights)
        
        # 2. Freeze the early layers of EfficientNet to leverage pre-trained knowledge
        # We will only fine-tune the last few blocks
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        for param in self.efficientnet.features[-3:].parameters():
            param.requires_grad = True
            
        # 3. Get the number of output features from EfficientNet's classifier
        num_features = self.efficientnet.classifier[1].in_features
        # Remove the original classifier
        self.efficientnet.classifier = nn.Identity()

        # 4. Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # 5. Define the final classifier
        lstm_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x shape: [B, T, C, H, W] where T is sequence length (in our case, T=1)
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w) # Reshape for EfficientNet
        
        # Get features from EfficientNet
        features = self.efficientnet(x)
        features = features.view(batch_size, seq_len, -1) # Reshape for LSTM
        
        # Pass features through LSTM
        lstm_out, _ = self.lstm(features)
        
        # We only need the output of the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Pass through the final classifier
        output = self.classifier(lstm_out)
        
        return output