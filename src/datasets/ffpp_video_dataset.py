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
            split (string): 'train', 'val', or 'test' to select the dataset split.
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

        # --- FIX: Load ALL manipulated video types (Label 1) ---
        manipulated_paths = []
        fake_types = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        
        print(f"Searching for manipulated video types: {fake_types}")

        for fake_type in fake_types:
            manipulated_root = os.path.join(self.root_dir, 'manipulated_sequences', fake_type)
            
            if not os.path.isdir(manipulated_root):
                 print(f"Warning: Directory not found, skipping - {manipulated_root}")
                 continue # Skip this fake type if the folder doesn't exist
            
            print(f"Loading from: {manipulated_root}")
            for dirpath, dirnames, filenames in os.walk(manipulated_root):
                 if os.path.basename(dirpath) == 'videos':
                    for filename in filenames:
                        if filename.lower().endswith(self.video_extensions):
                            manipulated_paths.append(os.path.join(dirpath, filename))
        # --- END OF FIX ---

        print(f"Found {len(original_paths)} original videos and {len(manipulated_paths)} manipulated videos.")

        # --- Create Train/Validation/Test Split ---
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
            
        elif self.split == 'test':
            # This implementation assumes the 'test' set is the same as the 'val' set
            # (i.e., the data not used for training)
            test_originals = original_paths[orig_split_idx:]
            test_manipulated = manipulated_paths[manip_split_idx:]
            self.samples.extend([(p, 0) for p in test_originals])
            self.samples.extend([(p, 1) for p in test_manipulated])
            
        else:
            raise ValueError(f"Invalid split name: {self.split}. Choose 'train', 'val', or 'test'.")

        # Shuffle combined list for good measure, though sampler handles train
        if self.split == 'train':
            random.shuffle(self.samples) 

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
            # Apply transform to each frame and stack them
            frames = torch.stack([self.transform(frame) for frame in frames])

        # Expected output shape: [num_frames, C, H, W]
        return frames, float(label)

    def _sample_frames(self, video_path):
        """Samples 'num_frames' evenly spaced frames from the video."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            # Return a placeholder black frame
            return [np.zeros((100, 100, 3), dtype=np.uint8)] * self.num_frames 

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1: # Handle empty or corrupt video
             print(f"Warning: Video has {total_frames} frames: {video_path}")
             return [np.zeros((100, 100, 3), dtype=np.uint8)] * self.num_frames

        # Ensure num_frames is not greater than total_frames
        num_to_sample = min(self.num_frames, total_frames)
        # Get evenly spaced indices
        indices = np.linspace(0, total_frames - 1, num_to_sample, dtype=int)

        frame_idx = 0
        sampled_count = 0
        processed_indices = set() # To avoid duplicates if indices are close

        while sampled_count < num_to_sample:
            ret = cap.grab() # Grab frame header
            if not ret:
                # End of video or error
                if frames:
                    # Pad with last good frame
                    while len(frames) < self.num_frames: frames.append(frames[-1])
                else:
                    # No frames read, pad with zeros
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
                ret, frame = cap.retrieve() # Decode the grabbed frame
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    processed_indices.add(target_idx)
                    sampled_count += 1
                else:
                    # Failed to retrieve frame
                    print(f"Warning: Could not retrieve frame {frame_idx} from {video_path}")
                    # Pad with last good frame if available, else pad with zeros
                    if frames: frames.append(frames[-1])
                    else: frames.append(np.zeros((100,100,3), dtype=np.uint8))
                    processed_indices.add(target_idx) # Mark as processed even if failed
                    sampled_count += 1

            frame_idx += 1
            # Optimization: If we've passed the last needed index, stop reading
            if frame_idx > indices[-1] and sampled_count >= num_to_sample:
                 break


        cap.release()

        # Pad if needed (e.g., if video was shorter than num_frames or read failed)
        while len(frames) < self.num_frames:
            if frames: frames.append(frames[-1]) # Pad with last frame
            else: frames.append(np.zeros((100,100,3), dtype=np.uint8)) # Pad with zeros

        # Ensure exactly num_frames are returned
        return frames[:self.num_frames]

