import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import warnings
import random

warnings.filterwarnings("ignore") # Ignore PIL warnings

class FFPPFrameDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, transform=None, split='train', 
                 train_ratio=0.8, val_ratio=0.1):
        """
        Args:
            root_dir (string): Root directory of the pre-processed FFPP frames 
                               (e.g., 'data/ffpp_frames').
            num_frames (int): Number of frames to sample from each video folder.
            transform (callable, optional): Optional transform to be applied on frames.
            split (string): 'train', 'val', or 'test'.
            train_ratio (float): Proportion of data for training (default: 0.8)
            val_ratio (float): Proportion of data for validation (default: 0.1)
                               Test ratio is inferred as (1.0 - train_ratio - val_ratio)
        """
        self.root_dir = os.path.join(root_dir, split)
        self.num_frames = num_frames
        self.transform = transform
        self.split = split
        self.samples = []

        print(f"Loading pre-processed frames from: {self.root_dir}")

        # --- THIS IS THE NEW, FIXED SPLITTING LOGIC ---
        # 1. Find all samples (video folders) for all types
        all_real_samples = self._find_samples(os.path.join(root_dir, 'original_sequences'))
        all_fake_samples = self._find_samples(os.path.join(root_dir, 'manipulated_sequences'))

        # 2. Combine and assign labels
        all_samples = [(sample, 0) for sample in all_real_samples] + \
                      [(sample, 1) for sample in all_fake_samples]
        
        # 3. Shuffle *before* splitting to ensure randomization
        random.seed(42) # Use a fixed seed for reproducible splits
        random.shuffle(all_samples)

        # 4. Calculate split indices
        total_samples = len(all_samples)
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)
        
        # 5. Assign samples based on the requested split
        if self.split == 'train':
            self.samples = all_samples[:train_end]
        elif self.split == 'val':
            self.samples = all_samples[train_end:val_end]
        elif self.split == 'test':
            self.samples = all_samples[val_end:]
        else:
            raise ValueError(f"Invalid split name: {self.split}. Choose 'train', 'val', or 'test'.")
        # --- END OF NEW SPLITTING LOGIC ---

        print(f"Loaded {len(self.samples)} samples for the '{self.split}' split.")
        if len(self.samples) == 0:
              raise RuntimeError(f"Found 0 videos for the '{self.split}' split. Check dataset structure and ratios.")


    def _find_samples(self, root_directory):
        """
        Helper function to find all video frame folders (samples)
        recursively within a given directory.
        A "sample" is a directory that contains frame_xx.jpg files.
        """
        samples = []
        if not os.path.isdir(root_directory):
            print(f"Warning: Directory not found, skipping: {root_directory}")
            return samples
            
        for dirpath, dirnames, filenames in os.walk(root_directory):
            # A "sample" dir is one that contains .jpg files directly
            if any(f.endswith('.jpg') for f in filenames):
                # Add the path to this directory (e.g., .../youtube/000_003)
                samples.append(dirpath)
                
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_folder_path, label = self.samples[idx]
        
        # Get all frame paths and sort them numerically
        try:
            all_frames = [os.path.join(video_folder_path, f) 
                          for f in os.listdir(video_folder_path) if f.endswith('.jpg')]
            
            # Sort frames by number (e.g., frame_0.jpg, frame_1.jpg...)
            all_frames.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
            
        except FileNotFoundError:
            print(f"Warning: Folder not found {video_folder_path}. Using a blank tensor.")
            return self._get_blank_tensor(), float(label)
        except Exception as e:
            print(f"Warning: Error reading {video_folder_path}: {e}. Using a blank tensor.")
            return self._get_blank_tensor(), float(label)

        total_frames = len(all_frames)
        if total_frames == 0:
            print(f"Warning: No frames found in {video_folder_path}. Using a blank tensor.")
            return self._get_blank_tensor(), float(label)

        # Sample 'num_frames' evenly spaced frames
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        sampled_frame_paths = [all_frames[i] for i in indices]

        frames = []
        for frame_path in sampled_frame_paths:
            try:
                # Open as PIL.Image (transforms expects this)
                frame = Image.open(frame_path).convert('RGB')
                frames.append(frame)
            except Exception as e:
                print(f"Warning: Could not load frame {frame_path}. Using last good frame or blank.")
                if frames: # Use last good frame
                    frames.append(frames[-1])
                else: # Or use a blank image if it's the first frame
                    frames.append(Image.new('RGB', (224, 224), (0, 0, 0)))

        # Apply transformations if provided (this should stack them)
        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])

        return frames, float(label)

    def _get_blank_tensor(self):
        """Returns a blank tensor matching the expected output shape."""
        blank_image = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform:
            blank_frame = self.transform(blank_image)
            return torch.stack([blank_frame] * self.num_frames)
        else:
            # Fallback if no transform
            return torch.zeros((self.num_frames, 3, 224, 224))

