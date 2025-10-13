# src/datasets/deepfake_dataset.py

import os
import cv2
from torch.utils.data import Dataset

class DeepfakeDataset(Dataset):
    # STEP 1: Add 'transform=None' to the __init__ method
    def __init__(self, data_dir, max_samples=None, transform=None):
        self.data_dir = data_dir
        self.samples = []
        self.transform = transform # Store the transform

        for label, sub_dir in enumerate(['real', 'fake']):
            class_path = os.path.join(data_dir, sub_dir)
            for file_name in os.listdir(class_path):
                self.samples.append((os.path.join(class_path, file_name), label))
        
        if max_samples:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image (assuming you use OpenCV)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB

        # STEP 2: Apply the transform if it exists
        if self.transform:
            image = self.transform(image)
        
        return image, float(label)