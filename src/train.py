import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder # <-- This is the key
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import warnings
from collections import Counter
import os

# --- 1. Import the NEW StaticImageModel ---
from src.models.static_image_model import StaticImageModel

warnings.filterwarnings("ignore")

# --- Configuration ---
CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 32, # Can use a larger batch size for images
    "num_workers": 4, 
    "learning_rate": 1e-4, 
    "epochs": 10, 
    "patience": 15,
    "data_dir": "data", # Root directory containing 'train' and 'val'
    "model_save_path": "best_static_image_model.pth" # New save path
}

# --- train_one_epoch and validate functions (simplified for images) ---
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0
    loop = tqdm(loader, leave=True, desc="Training")
    
    # --- Data loop is now simpler ---
    for images, labels in loop:
        images = images.to(CONFIG["device"])
        labels = labels.float().to(CONFIG["device"])

        with autocast(enabled=(CONFIG["device"].type == 'cuda')):
            outputs = model(images) # Pass 4D tensor
            loss = criterion(outputs.squeeze(-1), labels) 

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    loop = tqdm(loader, leave=True, desc="Validating")
    with torch.no_grad():
        for images, labels in loop:
            images = images.to(CONFIG["device"])
            labels = labels.float().to(CONFIG["device"])
            
            with autocast(enabled=(CONFIG["device"].type == 'cuda')):
                outputs = model(images) # Pass 4D tensor
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
    print(f"Loading data from: {CONFIG['data_dir']}")

    # --- 2. Define AGGRESSIVE transforms for static images ---
    # Static images are prone to overfitting, so we use strong augmentation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # --- 3. Use ImageFolder dataset ---
    # This assumes you have:
    # data/train/real/img1.png...
    # data/train/fake/img2.png...
    # data/val/real/img3.png...
    # data/val/fake/img4.png...
    try:
        train_dir = os.path.join(CONFIG["data_dir"], "train")
        val_dir = os.path.join(CONFIG["data_dir"], "val")
        
        train_set = ImageFolder(train_dir, transform=train_transforms)
        val_set   = ImageFolder(val_dir, transform=val_transforms)
    except FileNotFoundError as e:
        print(f"\n---!!! Dataset Loading Error !!!---")
        print(f"Error: {e}")
        print(f"Please ensure you have 'train' and 'val' folders inside '{CONFIG['data_dir']}'")
        print("And that they contain 'real' and 'fake' subfolders.")
        exit(1)

    # --- WeightedRandomSampler for Imbalance ---
    class_counts = Counter(train_set.targets)
    print(f"Training image class counts: {class_counts}")

    sampler = None
    if class_counts.get(0, 0) > 0 and class_counts.get(1, 0) > 0:
        print("Applying weighted random oversampler to training set.")
        # Class weights: class_counts[0] is 'fake', class_counts[1] is 'real'
        # Check train_set.class_to_idx to confirm
        print(f"Classes found: {train_set.class_to_idx}") # e.g., {'fake': 0, 'real': 1}
        
        # Make weights inversely proportional to class size
        # Get count for each class index (0, 1, ...)
        class_counts_list = [class_counts[i] for i in range(len(class_counts))]
        class_weights = [1.0 / count for count in class_counts_list]
        
        # Get weight for each sample
        sample_weights = [class_weights[label] for label in train_set.targets]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    else:
        print("Dataset is imbalanced or empty, not applying oversampling.")

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

    # --- 5. Instantiate the NEW StaticImageModel ---
    model = StaticImageModel().to(CONFIG["device"])
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    scaler = GradScaler(enabled=(CONFIG["device"].type == 'cuda'))

    # --- Training Loop ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print(f"\nStarting STATIC IMAGE training for {CONFIG['epochs']} epochs...")
    for epoch in range(CONFIG['epochs']):
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

