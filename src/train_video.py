import torch
import torch.nn as nn
from torch.utils.data import DataLoader # Removed WeightedRandomSampler for now
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import warnings
from collections import Counter

# Use the NEW Video Dataset
from src.datasets.video_deepfake_dataset import VideoDeepfakeDataset
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
    "num_frames": 16 # Number of frames to sample per video
}

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

    # --- Video Frame Transformations (Matching Previous Successful Training) ---
    video_transforms = transforms.Compose([
        transforms.ToTensor(), # Input to this is now a NumPy array (HWC) from OpenCV
        # transforms.Resize((224, 224), antialias=True), # Commented out
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Commented out
    ])

    # --- Datasets ---
    try:
        train_set = VideoDeepfakeDataset("data/train", num_frames=CONFIG["num_frames"], transform=video_transforms)
        val_set   = VideoDeepfakeDataset("data/val", num_frames=CONFIG["num_frames"], transform=video_transforms)
    except RuntimeError as e:
        print(f"\n---!!! Dataset Loading Error !!!---")
        print(f"Error: {e}")
        print("Please ensure your 'data/train' and 'data/val' folders contain 'real' and 'fake' subdirectories with video files (.mp4, .avi, .mov).")
        exit(1)

    # --- DataLoaders ---
    # NOTE: Oversampling (WeightedRandomSampler) is not included here initially.
    # Add it back if your video dataset is significantly imbalanced.
    # Check balance first:
    train_labels = [label for _, label in train_set.samples]
    print(f"Training video class counts: {Counter(train_labels)}")

    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)

    # --- Model, Loss, Optimizer ---
    model = EfficientNet_LSTM().to(CONFIG["device"])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    # --- Performance Boosters ---
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    scaler = GradScaler() # For Mixed Precision if using GPU

    # --- Training Loop with Early Stopping ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print(f"\nStarting video training for {CONFIG['epochs']} epochs...")
    for epoch in range(CONFIG["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the video-trained model with a different name
            torch.save(model.state_dict(), "best_video_model.pth")
            print(f"-> Validation loss improved. Saving model to best_video_model.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= CONFIG["patience"]:
            print(f"Early stopping triggered after {CONFIG['patience']} epochs with no improvement.")
            break

    print(f"Video training complete. Best model saved to best_video_model.pth")
      
