import torch
import cv2
import argparse
import json
import os
from torchvision import transforms
from PIL import Image # Use PIL for consistency if your dataset uses it

# 1. Import your new, more powerful model
from src.models.efficientnet_lstm import EfficientNet_LSTM
# 2. Import the updated Grad-CAM function
from src.explainability.gradcam import generate_gradcam
import warnings
warnings.filterwarnings("ignore") # Keep this if you need it

def predict_with_explain(frame_path, model_path, output_dir, threshold):
    """
    Runs a prediction on a single image, generates a Grad-CAM heatmap,
    and returns the decision.
    """
    
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 2. Load Model ---
    # Use the same EfficientNet_LSTM model class as in training
    model = EfficientNet_LSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- 3. Load and Transform Image ---
    # Use PIL Image open if your dataset uses it, otherwise stick to cv2
    img = Image.open(frame_path).convert("RGB") 
    
    # IMPORTANT: Use the exact same transformations as your validation set
    # Note: EfficientNet usually requires 224x224 and specific normalization
    transform = transforms.Compose([
        # transforms.Resize((224, 224)), # Resize if your EfficientNet expects it (B0 default is 224)
        transforms.ToTensor(),
        # Use ImageNet normalization if EfficientNet was pre-trained on it
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Or just ToTensor if that's what your validation set used:
        # transforms.Compose([ transforms.ToPILImage(), transforms.ToTensor() ]) 
    ])
    img_tensor = transform(img).to(device)

    # --- 4. Reshape Tensor ---
    # The model expects an input shape of [B, T, C, H, W]
    input_tensor = img_tensor.unsqueeze(0).unsqueeze(1) 

    # --- 5. Get Initial Prediction ---
    with torch.no_grad():
        output = model(input_tensor)
        # Use the raw output for GradCAM, apply sigmoid later for probability
        raw_output = output.item() 
        prob = torch.sigmoid(torch.tensor(raw_output)).item()

    # --- 6. Call Grad-CAM (Corrected) ---
    # Pass the raw probability (before sigmoid) if needed by gradcam logic, 
    # or the sigmoid probability (prob) if it expects a value between 0 and 1.
    # The updated gradcam function expects the sigmoid probability.
    final_prob_from_gradcam, heatmap_path = generate_gradcam(
        model, 
        input_tensor, 
        output_dir, 
        "gradcam.png", 
        prob # Pass the sigmoid probability
    )
    
    # It's usually better to trust the original model probability for the decision
    final_prob_to_use = prob 

    # --- 7. Make Decision ---
    decision = "FAKE" if final_prob_to_use > threshold else "REAL"
    
    # Ensure heatmap_path is not None before processing
    heatmap_display_path = "N/A"
    if heatmap_path:
        heatmap_display_path = heatmap_path.replace("\\", "/") # Use forward slashes

    report = {
        "decision": decision,
        "probability": final_prob_to_use,
        "heatmap": heatmap_display_path
    }
    
    return decision, final_prob_to_use, report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Detection")
    parser.add_argument('--frame', type=str, required=True, help="Path to the input image")
    parser.add_argument('--weights', type=str, required=True, help="Path to the trained model.pth file")
    parser.add_argument('--out', type=str, required=True, help="Directory to save the output heatmap")
    parser.add_argument('--threshold', type=float, default=0.5, help="Decision threshold for fake (default: 0.5)")
    
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.out, exist_ok=True)
    
    # Run the prediction and explanation
    try:
        decision, prob, report = predict_with_explain(args.frame, args.weights, args.out, args.threshold)
        # Print the final report as a clean JSON string
        print(json.dumps(report))
    except FileNotFoundError:
        print(f"Error: Input frame file not found at {args.frame}")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

