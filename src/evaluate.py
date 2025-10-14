import torch
import json
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# Import your custom modules
from src.models.cnn_lstm import CNN_LSTM
from src.datasets.deepfake_dataset2 import DeepfakeDataset

# --- Configuration ---
CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "weights_path": "best_model.pth",
    "data_dir": "data/val",
    "batch_size": 32,
    "num_workers": 2
}

def evaluate_model():
    """Loads a trained model and evaluates it on the validation dataset."""
    print(f"Using device: {CONFIG['device']}")

    # --- Load Model ---
    model = CNN_LSTM().to(CONFIG['device'])
    model.load_state_dict(torch.load(CONFIG['weights_path'], map_location=CONFIG['device']))
    model.eval()

    # --- Prepare Dataset ---
    val_transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    val_dataset = DeepfakeDataset(CONFIG['data_dir'], transform=val_transforms, return_path=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])

    # --- Run Predictions ---
    all_preds, all_labels, all_paths = [], [], []
    print("Running evaluation on the validation set...")
    with torch.no_grad():
        for frames, labels, paths in tqdm(val_loader):
            frames = frames.to(CONFIG['device'])
            outputs = model(frames.unsqueeze(1))
            preds = (torch.sigmoid(outputs).squeeze() > 0.5).int()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy().astype(int))
            all_paths.extend(paths)

    # --- Analyze and Display Results ---
    failures = []
    for i in range(len(all_paths)):
        if all_labels[i] != all_preds[i]:
            failures.append({
                "file": all_paths[i],
                "true_label": "FAKE" if all_labels[i] == 1 else "REAL",
                "predicted_label": "FAKE" if all_preds[i] == 1 else "REAL"
            })
    
    print("\n--- Evaluation Complete ---")
    if failures:
        print(f"\n🚨 Found {len(failures)} incorrect predictions. Saving to failures.json")
        with open("failures.json", 'w') as f:
            json.dump(failures, f, indent=4)
    else:
        print("\n✅ No incorrect predictions found!")

    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=['REAL', 'FAKE']))

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(all_labels, all_preds)
    print("         Predicted")
    print("         REAL  FAKE")
    print(f"Actual REAL: {cm[0][0]:>5} {cm[0][1]:>5}")
    print(f"Actual FAKE: {cm[1][0]:>5} {cm[1][1]:>5}")
    print("-" * 26)
    print(f"False Positives (Real predicted as Fake): {cm[0][1]}")
    print(f"False Negatives (Fake predicted as Real): {cm[1][0]}")

if __name__ == "__main__":
    evaluate_model()