import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import warnings
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- 1. USE THE NEW FRAME DATASET ---
from src.datasets.ffpp_frame_dataset import FFPPFrameDataset
from src.models.efficientnet_lstm import EfficientNet_LSTM

warnings.filterwarnings("ignore")

# --- Configuration ---
CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 16, # Use a larger batch size for faster evaluation
    "num_workers": 4,
    
    # --- 2. POINT TO NEW DATA/MODEL ---
    "ffpp_root_dir": "data/ffpp_frames", # Point to the pre-processed frames
    "num_frames": 16,
    "model_path": "best_video_model_fast.pth" # Load the new model
}

def evaluate_model(model, loader, criterion):
    """Evaluates the model on the test set."""
    model.eval()
    test_loss = 0
    all_labels = []
    all_preds = []
    
    loop = tqdm(loader, leave=True, desc="Evaluating")
    with torch.no_grad():
        for frame_sequences, labels in loop:
            frame_sequences, labels = frame_sequences.to(CONFIG["device"]), labels.float().to(CONFIG["device"])
            
            with torch.cuda.amp.autocast(enabled=(CONFIG["device"].type == 'cuda')):
                outputs = model(frame_sequences)
                loss = criterion(outputs.squeeze(-1), labels) 

            test_loss += loss.item()
            
            # Get predictions (0 or 1)
            preds = (torch.sigmoid(outputs) > 0.5).int().squeeze(-1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            loop.set_postfix(loss=loss.item())

    avg_loss = test_loss / len(loader) if len(loader) > 0 else 0
    
    # Calculate metrics
    report = classification_report(all_labels, all_preds, target_names=['Real (0)', 'Fake (1)'], output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, report, cm

if __name__ == "__main__":
    print(f"Using device: {CONFIG['device']}")
    print(f"Loading model from: {CONFIG['model_path']}")

    # --- 3. USE SIMPLER VALIDATION TRANSFORMS ---
    # Frames are PIL Images and ALREADY 224x224
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        # --- NO transforms.Resize() NEEDED! ---
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- 4. LOAD THE TEST SET WITH FFPPFrameDataset ---
    try:
        test_set = FFPPFrameDataset(
            CONFIG["ffpp_root_dir"], 
            num_frames=CONFIG["num_frames"], 
            transform=test_transforms, # Use test transforms
            split='test' # Use the 'test' split
        )
    except RuntimeError as e:
        print(f"\n---!!! Dataset Loading Error !!!---")
        print(f"Error: {e}")
        print(f"Please ensure your pre-processed 'test' frames directory is correct: {os.path.join(CONFIG['ffpp_root_dir'], 'test')}")
        exit(1)

    if len(test_set) == 0:
        print("Test set is empty. Exiting.")
        exit(1)

    test_loader = DataLoader(
        test_set,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )

    # --- Load Model and Criterion ---
    model = EfficientNet_LSTM().to(CONFIG["device"])
    try:
        model.load_state_dict(torch.load(CONFIG["model_path"], map_location=CONFIG["device"]))
    except FileNotFoundError:
        print(f"Error: Model file not found at {CONFIG['model_path']}")
        print("Please make sure you have trained the model and the file exists.")
        exit(1)
        
    criterion = nn.BCEWithLogitsLoss()

    # --- Run Evaluation ---
    print(f"Running evaluation on {len(test_set)} test samples...")
    test_loss, report, cm = evaluate_model(model, test_loader, criterion)

    print("\n--- Evaluation Complete ---")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {report['accuracy'] * 100:.2f}%")
    print("---")
    
    # Print metrics for the 'Fake' class (which we care about)
    fake_metrics = report.get('Fake (1)', {})
    print(f"Precision (Fake): {fake_metrics.get('precision', 0):.4f} (How many 'fake' predictions were correct?)")
    print(f"Recall (Fake):    {fake_metrics.get('recall', 0):.4f} (How many actual 'fakes' did we find?)")
    print(f"F1 Score (Fake):  {fake_metrics.get('f1-score', 0):.4f} (Balanced score for precision/recall)")
    print("---")

    # --- Plot Confusion Matrix ---
    print("Generating Confusion Matrix...")
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Real (0)', 'Predicted Fake (1)'],
                yticklabels=['Actual Real (0)', 'Actual Fake (1)'])
    plt.title('Test Set Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Save the plot
    save_path = "test_confusion_matrix_fast.png"
    plt.savefig(save_path)
    print(f"Confusion Matrix plot saved to {save_path}")
    # plt.show() # Disabled for non-interactive environments
