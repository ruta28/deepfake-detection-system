import os
import torch
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import warnings

def generate_gradcam(model, input_tensor, output_dir, file_name, prob):
    """Generates and saves a Grad-CAM heatmap."""
    
    model.eval()
    target_layer_name = 'efficientnet.features.8'
    warnings.filterwarnings("ignore", message="Using a non-default target layer")
    heatmap_path = os.path.join(output_dir, file_name)
    
    cam_extractor = None 
    
    try:
        cam_extractor = GradCAM(model, target_layer=target_layer_name)
        
        # We don't need class_idx here anymore for the calculation itself
        
        # Ensure requires_grad is true for input_tensor
        input_tensor.requires_grad_(True)
        
        # Run model forward pass to get scores
        scores = model(input_tensor) 
        
        # --- CORRECTED CALL (Attempt 3) ---
        # Explicitly target class index 0, as there's only one output neuron.
        cams = cam_extractor(class_idx=0, scores=scores) 
        # --- END OF CORRECTION ---

    except ValueError as e:
        print(f"Error initializing Grad-CAM: {e}")
        return prob, None 
    except Exception as e:
        print(f"An unexpected error occurred during Grad-CAM extraction: {e}")
        # Print the full traceback for detailed debugging
        import traceback
        traceback.print_exc() 
        return prob, None
    finally:
        # Cleanup hooks and grad requirement
        if cam_extractor is not None:
            cam_extractor.remove_hooks()
        if hasattr(input_tensor, 'requires_grad') and input_tensor.requires_grad:
           input_tensor.requires_grad_(False)

    if not cams or len(cams) == 0:
        print("Grad-CAM returned no output (cams list is empty).")
        return prob, None
        
    # Ensure heatmap tensor is valid before proceeding
    if not isinstance(cams[0], torch.Tensor) or cams[0].nelement() == 0:
        print("Grad-CAM output tensor is invalid or empty.")
        return prob, None
        
    heatmap = cams[0].squeeze(0).cpu() 
    original_image = to_pil_image(input_tensor.squeeze(0).squeeze(0).cpu()) 

    try:
        result = overlay_mask(original_image, to_pil_image(heatmap, mode='F'), alpha=0.5)
    except Exception as e:
        print(f"Error overlaying heatmap: {e}")
        import traceback
        traceback.print_exc()
        return prob, None

    try:
        os.makedirs(output_dir, exist_ok=True)
        result.save(heatmap_path)
    except Exception as e:
        print(f"Error saving heatmap: {e}")
        import traceback
        traceback.print_exc()
        return prob, None 

    return prob, heatmap_path
