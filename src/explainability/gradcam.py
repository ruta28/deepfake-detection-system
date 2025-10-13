# src/explainability/gradcam.py
import torch
from torchvision.transforms.functional import to_pil_image
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask

def generate_gradcam(model, input_tensor, target_layer, class_idx=None, save_path="gradcam_result.png"):
    """
    Generate and save Grad-CAM heatmap.
    """
    model.eval()
    with GradCAM(model, target_layer='cnn.feature_extractor.0.8.2') as cam_extractor:
        # forward
        # input_tensor: [3,224,224]
        out = model(input_tensor.unsqueeze(0).unsqueeze(1))  # [1,1,3,224,224]
        pred = torch.sigmoid(out).item()

        # pick predicted class if not provided
        class_idx = 1 if pred > 0.5 else 0 if class_idx is None else class_idx

        # activation map
        activation_map = cam_extractor(class_idx, out)

        # convert image + heatmap
        img = to_pil_image(input_tensor.cpu())
        heatmap = to_pil_image(activation_map[0].squeeze(0).cpu(), mode="F")

        # overlay and save
        result = overlay_mask(img, heatmap, alpha=0.5)
        result.save(save_path)

        return pred, save_path
