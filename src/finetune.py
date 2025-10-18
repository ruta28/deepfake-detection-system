import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Import your new, powerful model and the dataset class
from src.models.efficientnet_lstm import EfficientNet_LSTM
from src.datasets.deepfake_dataset2 import DeepfakeDataset

# --- Configuration ---
CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "base_model_path": "best_model.pth", # The expert model we are starting with
    "finetuned_model_path": "best_model_finetuned.pth", # The new model we will save
    "failures_json_path": "failures.json", # The list of hard examples
    "batch_size": 4, # Use a small batch size for this small dataset
    "epochs": 5, # Train for only a few epochs
    "learning_rate": 1e-6, # Use a VERY low learning rate for fine-tuning
}

def create_finetune_dataset(json_path, transform):
    """Creates a dataset containing only the failure cases."""
    with open(json_path, 'r') as f:
        failures = json.load(f)
    
    # Extract file paths and convert labels to numbers (REAL=0, FAKE=1)
    samples = []
    for item in failures:
        label = 1 if item['true_label'] == 'FAKE' else 0
        samples.append((item['file'], label))
        
    return DeepfakeDataset("data", transform=transform, custom_samples=samples)


if __name__ == "__main__":
    print(f"Using device: {CONFIG['device']}")

    # --- Load the Expert Model ---
    print(f"Loading base model from {CONFIG['base_model_path']}")
    model = EfficientNet_LSTM().to(CONFIG['device'])
    model.load_state_dict(torch.load(CONFIG['base_model_path'], map_location=CONFIG['device']))
    model.train() # Set the model to training mode

    # --- Create the "Hard Examples" Dataset ---
    print(f"Creating fine-tuning dataset from {CONFIG['failures_json_path']}")
    # Use the same aggressive augmentations as your last training run
    finetune_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))], p=0.5),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])
    finetune_dataset = create_finetune_dataset(CONFIG['failures_json_path'], finetune_transforms)
    finetune_loader = DataLoader(finetune_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # --- Setup for Fine-Tuning ---
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    
    print(f"Starting fine-tuning on {len(finetune_dataset)} failure cases for {CONFIG['epochs']} epochs...")
    for epoch in range(CONFIG['epochs']):
        total_loss = 0
        loop = tqdm(finetune_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for frames, labels in loop:
            frames, labels = frames.to(CONFIG['device']), labels.float().to(CONFIG['device'])
            
            # Standard training step
            outputs = model(frames.unsqueeze(1))
            loss = criterion(outputs.squeeze(), labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

    # --- Save the Fine-Tuned Model ---
    torch.save(model.state_dict(), CONFIG['finetuned_model_path'])
    print(f"\nFine-tuning complete. Model saved to {CONFIG['finetuned_model_path']}")