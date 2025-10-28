import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import warnings
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. IMPORT THE NEW TRANSFORMER MODEL ---
from src.models.efficientnet_transformer import EfficientNet_Transformer
from src.datasets.ffpp_frame_dataset import FFPPFrameDataset

warnings.filterwarnings("ignore")

# --- Configuration ---
CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 16, 
    "num_workers": 4,
    "ffpp_root_dir": "data/ffpp_frames", 
    "num_frames": 16,
    
    # --- 2. LOAD THE NEW TRANSFORMER MODEL ---
    "model_path": "best_transformer_model.pth" 
}

def evaluate_model(model, loader, criterion):
    """Evaluates the model and finds the best threshold."""
    model.eval()
    all_labels = []
    all_probs = [] # Store probabilities, not predictions
    
    loop = tqdm(loader, leave=True, desc="Evaluating")
    with torch.no_grad():
        for frame_sequences, labels in loop:
            frame_sequences, labels = frame_sequences.to(CONFIG["device"]), labels.float().to(CONFIG["device"])
            
            with torch.cuda.amp.autocast(enabled=(CONFIG["device"].type == 'cuda')):
                outputs = model(frame_sequences)
            
            probs = torch.sigmoid(outputs).squeeze(-1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # --- Find the Best F1-Score Threshold (Robust Method) ---
    thresholds = np.linspace(0.01, 0.99, 100)
    f1_scores = []

    for t in thresholds:
        preds = (all_probs > t).astype(int)
        f1 = f1_score(all_labels, preds, pos_label=1, zero_division=0)
        f1_scores.append(f1)
    
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]
    
    print(f"\n--- Threshold Analysis ---")
    
    # --- Get stats for the DEFAULT (0.5) threshold ---
    preds_0_5 = (all_probs > 0.5).astype(int)
    report_0_5 = classification_report(all_labels, preds_0_5, target_names=['Real (0)', 'Fake (1)'], output_dict=True, zero_division=0)
    default_f1 = report_0_5.get('Fake (1)', {}).get('f1-score', 0)
    print(f"Default (0.5) F1 Score: {default_f1:.4f}")
    
    print(f"Best F1 Score ({best_f1:.4f}) found at threshold: {best_threshold:.4f}")
    
    # --- Get stats for the NEW BEST threshold ---
    best_preds = (all_probs > best_threshold).astype(int)
    best_report = classification_report(all_labels, best_preds, target_names=['Real (0)', 'Fake (1)'], output_dict=True, zero_division=0)
    best_cm = confusion_matrix(all_labels, best_preds)
    
    return best_report, best_cm, report_0_5


if __name__ == "__main__":
    print(f"Using device: {CONFIG['device']}")
    print(f"Loading model from: {CONFIG['model_path']}")

    eval_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        test_set = FFPPFrameDataset(
            CONFIG["ffpp_root_dir"], 
            num_frames=CONFIG["num_frames"], 
            transform=eval_transforms,
            split='test'
        )
    except RuntimeError as e:
        print(f"\n---!!! Dataset Loading Error !!!---")
        print(f"Error: {e}")
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

    # --- 3. INSTANTIATE THE NEW TRANSFORMER MODEL ---
    model = EfficientNet_Transformer().to(CONFIG["device"])
    
    try:
        model.load_state_dict(torch.load(CONFIG["model_path"], map_location=CONFIG["device"]))
    except FileNotFoundError:
        print(f"Error: Model file not found at {CONFIG['model_path']}")
        exit(1)
    except RuntimeError as e:
        print(f"---!!! Model Load Error !!!---")
        print(f"Error: {e}")
        print("This can happen if you are trying to load an old (LSTM) model.")
        exit(1)
        
    criterion = nn.BCEWithLogitsLoss() 

    print(f"Running evaluation on {len(test_set)} test samples...")
    best_report, best_cm, report_0_5 = evaluate_model(model, test_loader, criterion)

    fake_metrics_0_5 = report_0_5.get('Fake (1)', {})
    fake_metrics_best = best_report.get('Fake (1)', {})

    print(f"\n--- Default (0.5) Threshold Results ---")
    print(f"Precision (Fake): {fake_metrics_0_5.get('precision', 0):.4f}")
    print(f"Recall (Fake):    {fake_metrics_0_5.get('recall', 0):.4f}")
    print(f"F1 Score (Fake):  {fake_metrics_0_5.get('f1-score', 0):.4f}")
    print("---")
    
    print(f"--- BEST Tuned Threshold Results ---")
    print(f"Test Accuracy: {best_report['accuracy'] * 100:.2f}%")
    print(f"Precision (Fake): {fake_metrics_best.get('precision', 0):.4f}")
    print(f"Recall (Fake):    {fake_metrics_best.get('recall', 0):.4f}")
    print(f"F1 Score (Fake):  {fake_metrics_best.get('f1-score', 0):.4f}")
    print("---")

    print("Generating Confusion Matrix for *Best* Threshold...")
    plt.figure(figsize=(10, 7))
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Real (0)', 'Predicted Fake (1)'],
                yticklabels=['Actual Real (0)', 'Actual Fake (1)'])
    plt.title('Test Set Confusion Matrix (Tuned Threshold)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    save_path = "test_transformer_confusion_matrix.png"
    plt.savefig(save_path)
    print(f"Confusion Matrix plot saved to {save_path}")
