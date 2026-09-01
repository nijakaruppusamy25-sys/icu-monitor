import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class Resp1DCNN(nn.Module):
    def __init__(self):
        super(Resp1DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)
        return self.regressor(x)

if __name__ == "__main__":
    DATA_PATH = "data/rr_realsense_dataset.pt"
    PRETRAINED_PATH = "models/best_resp_model.pth"
    SAVE_PATH = "models/Resp1DCNN_RealSense_FineTuned.pth"
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Compute Hardware: {device}")

    # 1. Load Augmented Dataset
    data = torch.load(DATA_PATH)
    samples, labels = data["samples"], data["labels"]
    
    # 80/20 Train-Val Split
    dataset_size = len(samples)
    train_size = int(0.85 * dataset_size)
    val_size = dataset_size - train_size
    
    full_dataset = TensorDataset(samples, labels)
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    print(f"Training on {train_size} samples | Validating on {val_size} samples")

    # 2. Model Setup
    model = Resp1DCNN().to(device)
    model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device))
    
    criterion = nn.L1Loss()  # Direct MAE loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    best_val_mae = float('inf')
    epochs = 50

    print("\n" + "=" * 55)
    print("      FINE-TUNING RESP1DCNN ON REALSENSE DATA")
    print("=" * 55)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            
            optimizer.zero_grad()
            preds = model(x_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(y_b)
            
        train_mae = train_loss / train_size
        scheduler.step()

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                val_preds = model(x_v)
                val_loss += criterion(val_preds, y_v).item() * len(y_v)
                
        val_mae = val_loss / val_size

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs}] | Train MAE: {train_mae:5.3f} BrPM | Val MAE: {val_mae:5.3f} BrPM")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), SAVE_PATH)

    print("=" * 55)
    print(f"Fine-Tuning Complete! Best Validation MAE: {best_val_mae:.3f} BrPM")
    print(f"Saved weights to {SAVE_PATH}")