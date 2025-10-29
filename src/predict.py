import torch
import torch.nn as nn
from torchvision import transforms
import argparse
import os
import cv2
import numpy as np
from PIL import Image

# --- Import our winning model ---
from src.models.efficientnet_lstm import EfficientNet_LSTM

# --- Configuration ---
CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "num_frames": 16,
    
    # --- Load our best model and threshold ---
    "model_path": "best_video_model_fast.pth", 
    "threshold": 0.4753 # The best F1-score threshold we found
}

def sample_frames(video_path, num_frames):
    """
    Samples 'num_frames' evenly spaced frames from a video file.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        print(f"Error: Video has {total_frames} frames: {video_path}")
        return None

    # Get 'num_frames' indices evenly spaced
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    processed_indices = set()
    frame_idx = 0
    sampled_count = 0

    while sampled_count < num_frames:
        ret = cap.grab()
        if not ret:
            break # End of video

        # Check if current frame index is one we need
        if frame_idx in indices and frame_idx not in processed_indices:
            ret, frame = cap.retrieve()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                processed_indices.add(frame_idx)
                sampled_count += 1
            else:
                print(f"Warning: Could not retrieve frame {frame_idx} from {video_path}")
        
        frame_idx += 1
        
        # Optimization
        if len(processed_indices) == num_frames:
            break

    cap.release()

    if len(frames) == 0:
        print("Error: No frames were sampled.")
        return None
    
    # Pad if we sampled fewer than num_frames (e.g., short video)
    while len(frames) < num_frames:
        frames.append(frames[-1]) # Duplicate the last frame

    return frames[:num_frames]

def predict_single_video(model, video_path, transforms):
    """
    Runs a prediction on a single video file.
    """
    model.eval()
    
    # 1. Sample frames from the video
    print(f"Sampling {CONFIG['num_frames']} frames from video...")
    frames = sample_frames(video_path, CONFIG['num_frames'])
    if frames is None:
        return
    
    # 2. Apply transformations
    frames_transformed = []
    for frame in frames:
        try:
            frame_pil = Image.fromarray(frame)
            frames_transformed.append(transforms(frame_pil))
        except Exception as e:
            print(f"Warning: Could not transform frame. Error: {e}")
            frames_transformed.append(torch.zeros(3, 224, 224))
            
    frames_tensor = torch.stack(frames_transformed).to(CONFIG['device'])
    # Add a batch dimension: (16, 3, 224, 224) -> (1, 16, 3, 224, 224)
    frames_tensor = frames_tensor.unsqueeze(0)

    # 3. Get model prediction
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(CONFIG["device"].type == 'cuda')):
            output = model(frames_tensor)
        
        # Squeeze the output and get the probability
        prob = torch.sigmoid(output).squeeze().item()

    # 4. Compare to our tuned threshold
    prediction = "FAKE" if prob > CONFIG['threshold'] else "REAL"
    confidence = prob if prediction == "FAKE" else 1 - prob

    print("\n--- Prediction Result ---")
    print(f"Model Output Score: {prob:.4f}")
    print(f"Decision Threshold: {CONFIG['threshold']}")
    print(f"\nFinal Prediction: ** {prediction} **")
    print(f"Confidence: {confidence*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict if a video is REAL or FAKE")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    args = parser.parse_args()

    print(f"Using device: {CONFIG['device']}")

    # 1. Define Transforms (must match validation set)
    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)), # Ensure frame is resized
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Load Model
    print(f"Loading model from {CONFIG['model_path']}...")
    model = EfficientNet_LSTM().to(CONFIG['device'])
    
    try:
        # Load the saved weights (make sure this file is in your root folder)
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
    except FileNotFoundError:
        print(f"Error: Model file not found at {CONFIG['model_path']}")
        print("Please make sure your 'best_lstm_model.pth' is in the main project directory.")
        exit(1)
    except RuntimeError as e:
        print(f"Error loading model weights: {e}")
        print("This can happen if the model architecture in src/models/efficientnet_lstm.py is different from the one used for training.")
        exit(1)

    # 3. Run Prediction
    if not os.path.exists(args.video):
        print(f"Error: Video file not found at {args.video}")
    else:
        predict_single_video(model, args.video, eval_transforms)
