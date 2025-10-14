# src/datasets/deepfake_dataset.py

import os
import cv2
from torch.utils.data import Dataset

class DeepfakeDataset(Dataset):
    # 1. Add 'return_path=False' to the __init__ method
    def __init__(self, data_dir, max_samples=None, transform=None, return_path=False):
        self.data_dir = data_dir
        self.samples = []
        self.transform = transform
        self.return_path = return_path # 2. Store the flag

        for label, sub_dir in enumerate(['real', 'fake']):
            class_path = os.path.join(data_dir, sub_dir)
            # Add a check to prevent errors if the directory doesn't exist
            if os.path.isdir(class_path):
                for file_name in os.listdir(class_path):
                    self.samples.append((os.path.join(class_path, file_name), label))

        if max_samples:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        # 3. Conditionally return the image path along with the image and label
        if self.return_path:
            return image, float(label), img_path
        else:
            return image, float(label)