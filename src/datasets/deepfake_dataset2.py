import os
import cv2
from torch.utils.data import Dataset

class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, max_samples=None, transform=None, return_path=False, custom_samples=None):
        self.transform = transform
        self.return_path = return_path

        # --- THIS IS THE CORRECTED LOGIC ---
        # If a custom list of samples is provided, use it and ignore data_dir.
        if custom_samples:
            self.samples = custom_samples
            self.data_dir = None # data_dir is not needed in this case
        
        # If no custom list is provided, build samples from the data_dir
        else:
            if data_dir is None:
                raise ValueError("data_dir cannot be None if custom_samples are not provided")
            
            self.data_dir = data_dir
            self.samples = []
            for label, sub_dir in enumerate(['real', 'fake']):
                class_path = os.path.join(data_dir, sub_dir)
                if os.path.isdir(class_path):
                    for file_name in os.listdir(class_path):
                        self.samples.append((os.path.join(class_path, file_name), label))

            if max_samples:
                self.samples = self.samples[:max_samples]
        # --- END OF CORRECTED LOGIC ---

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)
        
        if self.return_path:
            return image, float(label), img_path
        else:
            return image, float(label)