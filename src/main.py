# src/main.py
import os
import torch
from torchvision import transforms
from PIL import Image
from src.models.cnn_lstm import CNN_LSTM
from src.explainability.gradcam import generate_gradcam
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def predict_with_explain(video_frame, model_path, out_dir, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = CNN_LSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # preprocess input frame (just take one frame here for explainability)
    frame = Image.open(video_frame).convert("RGB")
    input_tensor = transform(frame).to(device)

    # Grad-CAM explainability
    pred, heatmap_path = generate_gradcam(
        model, input_tensor, target_layer="cnn.0", 
        save_path=os.path.join(out_dir, "gradcam.png")
    )

    decision = "FAKE" if pred > threshold else "REAL"
    report = {
        "decision": decision,
        "probability": float(pred),
        "heatmap": heatmap_path
    }

    return decision, pred, report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=str, required=True, help="path to extracted frame")
    parser.add_argument("--weights", type=str, required=True, help="trained model path")
    parser.add_argument("--out", type=str, required=True, help="output directory")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    decision, prob, report = predict_with_explain(args.frame, args.weights, args.out, args.threshold)
    print(report)
