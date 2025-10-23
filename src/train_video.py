import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler # Added Sampler back
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import warnings
from collections import Counter

# Use the NEW FFPP Video Dataset
from src.datasets.ffpp_video_dataset import FFPPVideoDataset
# Use the existing EfficientNet_LSTM model
from src.models.efficientnet_lstm import EfficientNet_LSTM

warnings.filterwarnings("ignore")

# --- Configuration ---
CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 8, # Reduced batch size due to memory constraints with video frames
    "num_workers": 2,
    "learning_rate": 1e-4, # Might need adjustment for video
    "epochs": 50,
    "patience": 5,
    "num_frames": 16, # Number of frames to sample per video
    "ffpp_root_dir": "data/ffpp" # Set the correct root directory
}

# --- train_one_epoch and validate functions remain the same ---
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    """Trains the model for one epoch using video frame sequences."""
    model.train()
    total_loss = 0
    loop = tqdm(loader, leave=True)
    for frame_sequences, labels in loop:
        # frame_sequences shape: [B, T, C, H, W] where T = num_frames
        frame_sequences, labels = frame_sequences.to(CONFIG["device"]), labels.float().to(CONFIG["device"])

        with autocast():
            # Model's forward method expects [B, T, C, H, W] directly
            outputs = model(frame_sequences)
            loss = criterion(outputs.squeeze(), labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # Optional: Gradient Clipping can help stabilize training with LSTMs
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)

def validate(model, loader, criterion):
    """Validates the model using video frame sequences."""
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for frame_sequences, labels in loader:
            frame_sequences, labels = frame_sequences.to(CONFIG["device"]), labels.float().to(CONFIG["device"])
            with autocast():
                outputs = model(frame_sequences)
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

    # --- Video Frame Transformations (Matching Previous Successful Image Training - ONLY ToTensor) ---
    video_transforms = transforms.Compose([
        transforms.ToTensor(), # Input to this is now a NumPy array (HWC) from OpenCV
        # transforms.Resize((224, 224), antialias=True), # Commented out
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Commented out
    ])

    # --- Datasets (Using FFPPVideoDataset) ---
    try:
        # Pass the correct root directory and specify 'train' split
        train_set = FFPPVideoDataset(CONFIG["ffpp_root_dir"], num_frames=CONFIG["num_frames"], transform=video_transforms, split='train')
        # Pass the correct root directory and specify 'val' split
        val_set   = FFPPVideoDataset(CONFIG["ffpp_root_dir"], num_frames=CONFIG["num_frames"], transform=video_transforms, split='val')
    except RuntimeError as e:
        print(f"\n---!!! Dataset Loading Error !!!---")
        print(f"Error: {e}")
        print(f"Please ensure your '{CONFIG['ffpp_root_dir']}' folder contains the expected FFPP structure (original_sequences, manipulated_sequences).")
        exit(1)
    except FileNotFoundError as e:
        print(f"\n---!!! Dataset Loading Error !!!---")
        print(f"Error: {e}")
        print(f"Could not find the root directory specified: '{CONFIG['ffpp_root_dir']}'. Please check the path.")
        exit(1)


    # --- Check for Imbalance & Apply Oversampling if needed ---
    train_labels = [label for _, label in train_set.samples]
    class_counts = Counter(train_labels)
    print(f"Training video class counts: {class_counts}")

    sampler = None # Initialize sampler to None
    # Example: Apply oversampling if one class is less than 80% the size of the other
    majority_count = max(class_counts.values()) if class_counts else 0
    minority_count = min(class_counts.values()) if class_counts else 0
    if majority_count > 0 and minority_count > 0 and minority_count < 0.8 * majority_count:
        print("Applying oversampling due to class imbalance.")
        class_weights = {label: 1.0 / count for label, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    else:
        print("Dataset is reasonably balanced or empty, not applying oversampling.")


    # --- DataLoaders ---
    # Use sampler if defined, otherwise shuffle=True
    train_loader = DataLoader(
        train_set,
        batch_size=CONFIG["batch_size"],
        sampler=sampler, # Pass sampler here
        shuffle=(sampler is None), # Only shuffle if sampler is not used
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        # persistent_workers=True if CONFIG["num_workers"] > 0 else False # Can speed up loading
    )
    val_loader   = DataLoader(
        val_set,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        # persistent_workers=True if CONFIG["num_workers"] > 0 else False
    )

    # --- Model, Loss, Optimizer ---
    model = EfficientNet_LSTM().to(CONFIG["device"])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    # --- Performance Boosters ---
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    # Ensure scaler is only enabled when using CUDA
    scaler = GradScaler(enabled=(CONFIG["device"].type == 'cuda'))

    # --- Training Loop with Early Stopping ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print(f"\nStarting FFPP video training for {CONFIG['epochs']} epochs...")
    for epoch in range(CONFIG["epochs"]):
        # Pass scaler correctly, even if on CPU (it will be disabled)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        # Pass scaler to validate function if autocast is used there too
        val_loss, val_acc = validate(model, val_loader, criterion) # Assuming validate doesn't use scaler
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_video_model.pth")
            print(f"-> Validation loss improved. Saving model to best_video_model.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= CONFIG["patience"]:
            print(f"Early stopping triggered after {CONFIG['patience']} epochs with no improvement.")
            break

    print(f"Video training complete. Best model saved to best_video_model.pth")

