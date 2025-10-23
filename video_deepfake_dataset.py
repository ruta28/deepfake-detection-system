import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import warnings

warnings.filterwarnings("ignore") # OpenCV can be noisy

class VideoDeepfakeDataset(Dataset):
    def __init__(self, data_dir, num_frames=16, transform=None):
        """
        Args:
            data_dir (string): Directory with 'real' and 'fake' subdirectories containing videos.
            num_frames (int): Number of frames to sample from each video.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.transform = transform
        self.samples = []
        self.video_extensions = ('.mp4', '.avi', '.mov', '.mkv') # Add more if needed

        print(f"Loading video paths from: {self.data_dir}")
        for label, sub_dir in enumerate(['real', 'fake']):
            class_path = os.path.join(data_dir, sub_dir)
            if not os.path.isdir(class_path):
                print(f"Warning: Directory not found - {class_path}")
                continue

            for file_name in sorted(os.listdir(class_path)):
                if file_name.lower().endswith(self.video_extensions):
                    self.samples.append((os.path.join(class_path, file_name), label))

        print(f"Found {len(self.samples)} video samples.")
        if len(self.samples) == 0:
             raise RuntimeError(f"Found 0 videos in {self.data_dir}. Check dataset structure and video extensions.")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._sample_frames(video_path)

        # Apply transformations if provided
        if self.transform:
            # Apply transform to each frame individually
            frames = torch.stack([self.transform(frame) for frame in frames])

        # Expected output shape: [num_frames, C, H, W]
        return frames, float(label)

    def _sample_frames(self, video_path):
        """Samples 'num_frames' evenly spaced frames from the video."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            # Return placeholder frames or raise error
            # Returning black frames of arbitrary size for now
            return [np.zeros((100, 100, 3), dtype=np.uint8)] * self.num_frames 

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate indices of frames to sample
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frame_idx = 0
        sampled_count = 0
        
        while sampled_count < self.num_frames:
            ret = cap.grab() # Grab frame without decoding (faster)
            if not ret:
                 # If video ends early, duplicate the last frame
                 if frames:
                     while len(frames) < self.num_frames:
                         frames.append(frames[-1])
                 else: # Video couldn't be read at all
                     print(f"Error reading frames from {video_path}")
                     frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * self.num_frames
                 break

            if frame_idx in indices:
                ret, frame = cap.retrieve() # Decode the grabbed frame
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    sampled_count += 1
                else:
                    print(f"Warning: Could not retrieve frame {frame_idx} from {video_path}")
                    # If retrieve fails, try duplicating last good frame
                    if frames: frames.append(frames[-1])
                    else: frames.append(np.zeros((100,100,3), dtype=np.uint8)) # Fallback if first frame fails
                    sampled_count += 1 # Count it to ensure we get num_frames

            frame_idx += 1
        
        cap.release()

        # Handle cases where sampling might slightly overshoot due to rounding/retrieval issues
        if len(frames) > self.num_frames:
            frames = frames[:self.num_frames]
        elif len(frames) < self.num_frames: # Pad if still too short
            while len(frames) < self.num_frames:
                if frames: frames.append(frames[-1])
                else: frames.append(np.zeros((100,100,3), dtype=np.uint8))


        return frames
