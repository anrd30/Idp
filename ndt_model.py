import torch
import torch.nn as nn

class NDTNet(nn.Module):
    def __init__(self):
        super(NDTNet, self).__init__()
        # Encoder: 1D CNN to extract features from signals
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8) # Output: (batch, 256, 8)
        )
        
        # Bottleneck / Latent bridge
        self.fc_bridge = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128 * 5 * 5),
            nn.ReLU()
        )
        
        # Decoder: Transposed Convolution to reconstruct 50x50 image
        self.decoder = nn.Sequential(
            # Start from 5x5
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 10x10
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=1), # 21x21
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # 42x42
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=11, stride=1, padding=1) # Target is 50x50, adjust slightly
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc_bridge(x)
        x = x.view(-1, 128, 5, 5)
        x = self.decoder(x)
        x = torch.nn.functional.interpolate(x, size=(50, 50), mode='bilinear', align_corners=False)
        return torch.sigmoid(x)
