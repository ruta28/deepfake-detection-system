import os
import cv2
import numpy as np
from tqdm import tqdm
import warnings
from src.datasets.ffpp_video_dataset import FFPPVideoDataset # Use the logic from our existing dataset

# --- CONFIGURATION ---
FFPP_ROOT = "data/ffpp"
OUTPUT_ROOT = "data/ffpp_frames"
NUM_FRAMES = 16
IMG_SIZE = (224, 224) # The target size your model expects
# --- END CONFIGURATION ---

def process_and_save_frames(dataset, split_name):
    """
    Loops through a dataset split, extracts, resizes, and saves frames.
    """
    output_split_dir = os.path.join(OUTPUT_ROOT, split_name)
    print(f"\nProcessing split: '{split_name}'. Outputting to: {output_split_dir}")
    
    # Use tqdm for a nice progress bar
    for video_path, label in tqdm(dataset.samples, desc=f"Processing {split_name}"):
        try:
            # 1. Create the new directory structure
            label_str = "fake" if label == 1 else "real"
            
            # Create a unique video ID from its path
            # e.g., .../Deepfakes/c23/videos/000_003.mp4 -> Deepfakes_000_003
            # e.g., .../youtube/c23/videos/000.mp4 -> youtube_000
            video_name = os.path.basename(video_path).split('.')[0]
            source_type = video_path.split(os.sep)[-4] # e.g., Deepfakes or youtube
            unique_video_id = f"{source_type}_{video_name}"
            
            video_frame_dir = os.path.join(output_split_dir, label_str, unique_video_id)
            os.makedirs(video_frame_dir, exist_ok=True)
            
            # 2. Check if frames already exist to make this script resumable
            if len(os.listdir(video_frame_dir)) == NUM_FRAMES:
                continue

            # 3. Use the dataset's sampling logic to get frame arrays (as RGB)
            frames_rgb = dataset._sample_frames(video_path)
            
            if not frames_rgb or len(frames_rgb) == 0:
                print(f"Warning: Failed to sample frames for {video_path}")
                continue

            # 4. Resize and save each frame
            for i, frame_rgb in enumerate(frames_rgb):
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                frame_resized = cv2.resize(frame_bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)
                
                frame_filename = f"frame_{i:02d}.jpg"
                frame_save_path = os.path.join(video_frame_dir, frame_filename)
                
                # Save as JPEG (quality 95 is high)
                cv2.imwrite(frame_save_path, frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                
        except Exception as e:
            print(f"\n--- Error processing {video_path} ---")
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print("Starting video frame pre-processing...")
    print(f"Source video directory: {FFPP_ROOT}")
    print(f"Target frame directory: {OUTPUT_ROOT}")

    # We will instantiate our original dataset just to use its file-finding logic
    # We pass 'None' for transform because we are doing our own resizing
    try:
        train_dataset = FFPPVideoDataset(FFPP_ROOT, num_frames=NUM_FRAMES, transform=None, split='train')
        val_dataset = FFPPVideoDataset(FFPP_ROOT, num_frames=NUM_FRAMES, transform=None, split='val')
        test_dataset = FFPPVideoDataset(FFPP_ROOT, num_frames=NUM_FRAMES, transform=None, split='test')
        
        # Run the processing
        process_and_save_frames(train_dataset, 'train')
        process_and_save_frames(val_dataset, 'val')
        process_and_save_frames(test_dataset, 'test')
        
        print("\n--- Pre-processing Complete ---")
        print(f"All frames have been extracted and saved to {OUTPUT_ROOT}")

    except Exception as e:
        print(f"\nAn error occurred during dataset initialization: {e}")
        print("Please ensure 'src/datasets/ffpp_video_dataset.py' exists and the path 'data/ffpp' is correct.")
