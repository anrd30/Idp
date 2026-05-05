import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import glob
import os

def load_data(data_dir, max_samples=100000):
    print("Loading dataset into RAM...")
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    files.sort()
    
    all_x = []
    all_y = []
    
    count = 0
    for f in files:
        data = np.load(f)
        all_x.append(data['X'])
        all_y.append(data['Y'])
        count += data['X'].shape[0]
        if count >= max_samples:
            break
    
    if not all_x:
        raise ValueError("No data found in directory: " + data_dir)
        
    X = np.concatenate(all_x, axis=0)[:max_samples]
    Y = np.concatenate(all_y, axis=0)[:max_samples]
    
    X_tensor = torch.from_numpy(X).float()
    Y_tensor = torch.from_numpy(Y).float().unsqueeze(1)
    
    print(f"Loaded {X.shape[0]} samples.")
    return TensorDataset(X_tensor, Y_tensor)

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
        # Note: Getting exactly 50x50 with ConvTranspose can be tricky, using interpolation for safety
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc_bridge(x)
        x = x.view(-1, 128, 5, 5)
        x = self.decoder(x)
        x = torch.nn.functional.interpolate(x, size=(50, 50), mode='bilinear', align_corners=False)
        return torch.sigmoid(x)

def weighted_mse_loss(output, target, weight_factor=10.0):
    # Reduced weight factor for better stability
    mse = (output - target)**2
    weights = 1.0 + (target > 0.05).float() * weight_factor
    return (mse * weights).mean()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    try:
        # Load a bit more if possible, but keep 100k as base
        dataset = load_data("cnn_dataset", max_samples=100000)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    
    model = NDTNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(50): # Increased max epochs
        model.train()
        train_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = weighted_mse_loss(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if (i+1) % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item():.6f}")
                
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                val_loss += weighted_mse_loss(output, y).item()
        
        avg_val_loss = val_loss/len(val_loader)
        print(f"--- Epoch {epoch+1} Summary ---")
        print(f"Train Loss: {train_loss/len(train_loader):.6f}")
        print(f"Val Loss: {avg_val_loss:.6f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "ndt_cnn_model.pth")
            print("Best model saved.")
            
        # Early exit if loss is very low
        if avg_val_loss < 0.01:
            print("Converged!")
            break
        
    print(f"Final best val loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    train()
