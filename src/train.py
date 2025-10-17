import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import warnings
from collections import Counter

from src.models.cnn_lstm import CNN_LSTM
# Make sure this import matches your dataset's filename
from src.datasets.deepfake_dataset import DeepfakeDataset

warnings.filterwarnings("ignore")

# --- Configuration ---
CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 32,
    "num_workers": 4,
    "learning_rate": 1e-4,
    "epochs": 50,
    "patience": 5,
}

def train_one_epoch(model, loader, optimizer, criterion, scaler):
    # This function remains the same
    model.train()
    total_loss = 0
    loop = tqdm(loader, leave=True)
    for frames, labels in loop:
        frames, labels = frames.to(CONFIG["device"]), labels.float().to(CONFIG["device"])
        with autocast():
            outputs = model(frames.unsqueeze(1))
            loss = criterion(outputs.squeeze(), labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)

def validate(model, loader, criterion):
    # This function remains the same
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for frames, labels in loader:
            frames, labels = frames.to(CONFIG["device"]), labels.float().to(CONFIG["device"])
            with autocast():
                outputs = model(frames.unsqueeze(1))
                loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).int()
            correct += (preds.squeeze() == labels.int()).sum().item()
            total += labels.size(0)
    avg_loss = val_loss / len(loader) if len(loader) > 0 else 1
    accuracy = 100 * correct / total if total > 0 else 0
    return avg_loss, accuracy


if __name__ == "__main__":
    print(f"Using device: {CONFIG['device']}")

    # ==============================================================================
    #  NEW: AGGRESSIVE DATA AUGMENTATION
    # ==============================================================================
    # These transformations simulate real-world video artifacts like compression
    # and blur, forcing the model to learn more robust features.
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        # Apply a set of color/brightness adjustments with a 50% probability
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ], p=0.5),
        # Apply Gaussian blur with a 50% probability
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))
        ], p=0.5),
        transforms.ToTensor(), # Converts to tensor
        # This simulates compression artifacts by randomly erasing parts of the image
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
    ])
    # ==============================================================================
    
    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    # --- Datasets ---
    train_set = DeepfakeDataset("data/train", transform=train_transforms)
    val_set   = DeepfakeDataset("data/val", transform=val_transforms)

    # --- Oversampling Logic (remains the same) ---
    print("Implementing oversampling to handle data imbalance...")
    labels = [label for _, label in train_set.samples]
    class_counts = Counter(labels)
    print(f"Original training set class counts: {class_counts}")
    class_weights = {label: 1.0 / count for label, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # --- DataLoaders ---
    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"], sampler=sampler, num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)

    # --- Model, Loss, Optimizer (remains the same) ---
    model = CNN_LSTM().to(CONFIG["device"])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    
    # --- Performance Boosters (remains the same) ---
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    scaler = GradScaler()
    
    # --- Training Loop (remains the same) ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(CONFIG["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"-> Validation loss improved. Saving model to best_model.pth")
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= CONFIG["patience"]:
            print(f"Early stopping triggered after {CONFIG['patience']} epochs with no improvement.")
            break
    print("Training complete. Best model saved to best_model.pth")

