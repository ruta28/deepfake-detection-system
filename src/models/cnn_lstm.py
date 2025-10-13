import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

class FrameCNN(nn.Module):
    def __init__(self):
        super().__init__()
        base = efficientnet_b0(weights="DEFAULT")
        self.feature_extractor = nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1))
        self.out_dim = 1280

    def forward(self, x):  # x: [B,3,224,224]
        f = self.feature_extractor(x)
        return f.view(f.size(0), -1)

class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = FrameCNN()
        self.lstm = nn.LSTM(self.cnn.out_dim, 256, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):  # x: [B,T,3,224,224]
        B,T,C,H,W = x.size()
        x = x.view(B*T, C, H, W)
        feats = self.cnn(x)
        feats = feats.view(B, T, -1)
        out,_ = self.lstm(feats)
        hT = out[:,-1,:]
        return self.fc(hT)
