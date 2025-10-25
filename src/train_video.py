import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import warnings
from collections import Counter
import os

# --- USE THE FAST FRAME DATASET ---
from src.datasets.ffpp_frame_dataset import FFPPFrameDataset
from src.models.efficientnet_lstm import EfficientNet_LSTM # Use the original (no dropout)

warnings.filterwarnings("ignore")

# --- Configuration for our BEST Model ---
CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 8,
    "num_workers": 8,
    
    # --- Champion settings ---
    "learning_rate": 1e-4, 
    "epochs": 50,     
    "patience": 15,   
    
    "num_frames": 16,
    "ffpp_root_dir": "data/ffpp_frames", 
    "model_save_path": "best_video_model_fast.pth"
}

# --- train_one_epoch and validate functions (no changes) ---
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0
    loop = tqdm(loader, leave=True, desc="Training")
    for frame_sequences, labels in loop:
        frame_sequences, labels = frame_sequences.to(CONFIG["device"]), labels.float().to(CONFIG["device"])

        with autocast(enabled=(CONFIG["device"].type == 'cuda')):
            outputs = model(frame_sequences)
            loss = criterion(outputs.squeeze(-1), labels) 

        optimizer.zero_grad()
        if CONFIG["device"].type == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    loop = tqdm(loader, leave=True, desc="Validating")
    with torch.no_grad():
        for frame_sequences, labels in loop:
            frame_sequences, labels = frame_sequences.to(CONFIG["device"]), labels.float().to(CONFIG["device"])
            with autocast(enabled=(CONFIG["device"].type == 'cuda')):
                outputs = model(frame_sequences)
                loss = criterion(outputs.squeeze(-1), labels) 

            val_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).int().squeeze(-1)
            correct += (preds == labels.int()).sum().item()
            total += labels.size(0)

    avg_loss = val_loss / len(loader) if len(loader) > 0 else 1
    accuracy = 100 * correct / total if total > 0 else 0
    return avg_loss, accuracy


if __name__ == "__main__":
    print(f"Using device: {CONFIG['device']}")
    print(f"Loading data from: {CONFIG['ffpp_root_dir']}")

    # --- Use SIMPLE Augmentations (from our best model) ---
    train_transforms = transforms.Compose([
        # Standard augmentations that led to our best score
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Simple color change
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- Datasets ---
    try:
        train_set = FFPPFrameDataset(CONFIG["ffpp_root_dir"], num_frames=CONFIG["num_frames"], transform=train_transforms, split='train')
        val_set   = FFPPFrameDataset(CONFIG["ffpp_root_dir"], num_frames=CONFIG["num_frames"], transform=val_transforms, split='val')
    except RuntimeError as e:
        print(f"\n---!!! Dataset Loading Error !!!---")
        print(f"Error: {e}")
        exit(1)


    # --- Oversampling ---
    train_labels = [label for _, label in train_set.samples]
    class_counts = Counter(train_labels)
    print(f"Training frame class counts: {class_counts}")

    sampler = None
    if class_counts and class_counts[0] > 0 and class_counts[1] > 0:
        print("Applying weighted random oversampler to training set.")
        class_weights = {label: 1.0 / count for label, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    else:
        print("Dataset is imbalanced or one class is missing. Not applying oversampling.")

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_set,
        batch_size=CONFIG["batch_size"],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )
    val_loader   = DataLoader(
        val_set,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )

    # --- Model, Loss, Optimizer (AdamW) ---
    model = EfficientNet_LSTM().to(CONFIG["device"])
    criterion = nn.BCEWithLogitsLoss()
    
    # --- Use AdamW (our champion optimizer) ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    # --- Schedulers ---
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7) 
    scaler = GradScaler(enabled=(CONFIG["device"].type == 'cuda'))

    # --- Training Loop ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print(f"\nStarting FAST training for {CONFIG['epochs']} epochs...")
    for epoch in range(CONFIG["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print(f"-> Validation loss improved. Saving model to {CONFIG['model_save_path']}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= CONFIG["patience"]:
            print(f"Early stopping triggered after {CONFIG['patience']} epochs with no improvement.")
            break

    print(f"Fast training complete. Best model saved to {CONFIG['model_save_path']}")

