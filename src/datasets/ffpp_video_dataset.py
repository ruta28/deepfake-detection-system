import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import warnings
import random

warnings.filterwarnings("ignore") # OpenCV can be noisy

class FFPPVideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, transform=None, split='train', ratio=0.8):
        """
        Args:
            root_dir (string): Root directory of the FFPP dataset (e.g., 'data/ffpp').
            num_frames (int): Number of frames to sample from each video.
            transform (callable, optional): Optional transform to be applied on frames.
            split (string): 'train' or 'val' to select the dataset split.
            ratio (float): Train/validation split ratio (default: 0.8 for train).
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.split = split
        self.video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        self.samples = []

        print(f"Loading FFPP video paths from: {self.root_dir} for split: {self.split}")

        # Find all original videos (Label 0)
        original_paths = []
        for subdir in ['original_sequences/youtube', 'original_sequences/actors']:
            path = os.path.join(self.root_dir, subdir)
            if not os.path.isdir(path):
                print(f"Warning: Original sequences directory not found - {path}")
                continue
            # Walk through c23, c40 etc. then videos folder
            for dirpath, dirnames, filenames in os.walk(path):
                if os.path.basename(dirpath) == 'videos':
                    for filename in filenames:
                        if filename.lower().endswith(self.video_extensions):
                            original_paths.append(os.path.join(dirpath, filename))

        # Find all manipulated videos (Label 1) - focusing on Deepfakes for now
        manipulated_paths = []
        manipulated_root = os.path.join(self.root_dir, 'manipulated_sequences/Deepfakes')
        if not os.path.isdir(manipulated_root):
             print(f"Warning: Manipulated sequences directory not found - {manipulated_root}")
        else:
            for dirpath, dirnames, filenames in os.walk(manipulated_root):
                 if os.path.basename(dirpath) == 'videos':
                    for filename in filenames:
                        if filename.lower().endswith(self.video_extensions):
                            manipulated_paths.append(os.path.join(dirpath, filename))

        print(f"Found {len(original_paths)} original videos and {len(manipulated_paths)} manipulated videos.")

        # --- Create Train/Validation Split ---
        random.seed(42) # for reproducible splits
        random.shuffle(original_paths)
        random.shuffle(manipulated_paths)

        orig_split_idx = int(len(original_paths) * ratio)
        manip_split_idx = int(len(manipulated_paths) * ratio)

        if self.split == 'train':
            train_originals = original_paths[:orig_split_idx]
            train_manipulated = manipulated_paths[:manip_split_idx]
            self.samples.extend([(p, 0) for p in train_originals])
            self.samples.extend([(p, 1) for p in train_manipulated])
        elif self.split == 'val':
            val_originals = original_paths[orig_split_idx:]
            val_manipulated = manipulated_paths[manip_split_idx:]
            self.samples.extend([(p, 0) for p in val_originals])
            self.samples.extend([(p, 1) for p in val_manipulated])
        else:
            raise ValueError(f"Invalid split name: {self.split}. Choose 'train' or 'val'.")

        random.shuffle(self.samples) # Shuffle combined list

        print(f"Loaded {len(self.samples)} samples for the '{self.split}' split.")
        if len(self.samples) == 0:
             raise RuntimeError(f"Found 0 videos for the '{self.split}' split. Check dataset structure.")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._sample_frames(video_path)

        # Apply transformations if provided
        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])

        # Expected output shape: [num_frames, C, H, W]
        return frames, float(label)

    def _sample_frames(self, video_path):
        """Samples 'num_frames' evenly spaced frames from the video."""
        # --- Frame sampling logic remains the same as VideoDeepfakeDataset ---
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return [np.zeros((100, 100, 3), dtype=np.uint8)] * self.num_frames # Placeholder

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1: # Handle empty or corrupt video
             print(f"Warning: Video has {total_frames} frames: {video_path}")
             return [np.zeros((100, 100, 3), dtype=np.uint8)] * self.num_frames

        # Ensure num_frames is not greater than total_frames
        num_to_sample = min(self.num_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, num_to_sample, dtype=int)

        frame_idx = 0
        sampled_count = 0
        processed_indices = set() # To avoid duplicates if indices are close

        while sampled_count < num_to_sample:
            ret = cap.grab()
            if not ret:
                 if frames:
                     while len(frames) < self.num_frames: frames.append(frames[-1])
                 else:
                     frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * self.num_frames
                 break

            # Check if current frame index is one we need to sample
            is_target_frame = False
            target_idx = -1
            for i in indices:
                if frame_idx == i and i not in processed_indices:
                    is_target_frame = True
                    target_idx = i
                    break

            if is_target_frame:
                ret, frame = cap.retrieve()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    processed_indices.add(target_idx)
                    sampled_count += 1
                else:
                    print(f"Warning: Could not retrieve frame {frame_idx} from {video_path}")
                    if frames: frames.append(frames[-1])
                    else: frames.append(np.zeros((100,100,3), dtype=np.uint8))
                    processed_indices.add(target_idx) # Mark as processed even if failed
                    sampled_count += 1

            frame_idx += 1
            # Optimization: If we've passed the last needed index, stop reading
            if frame_idx > indices[-1] and sampled_count >= num_to_sample:
                 break


        cap.release()

        # Pad if needed (e.g., if video was shorter than num_frames)
        while len(frames) < self.num_frames:
            if frames: frames.append(frames[-1])
            else: frames.append(np.zeros((100,100,3), dtype=np.uint8))

        # Ensure exactly num_frames are returned
        return frames[:self.num_frames]
