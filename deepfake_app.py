import os
import io
import uuid
import warnings
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
# Ensure scikit-image is installed: pip install scikit-image
from skimage.transform import resize

from flask import Flask, request, jsonify, render_template_string, send_from_directory

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Using a non-default target layer")
warnings.filterwarnings("ignore", message="Unable to retrieve source code") # Common torchcam warning


# --- Configuration ---
MODEL_WEIGHTS_PATH = "best_model_finetuned.pth"
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- EfficientNet_LSTM Model Definition ---
class EfficientNet_LSTM(nn.Module):
    # --- Model class definition remains the same ---
    def __init__(self, lstm_hidden_size=256, lstm_layers=2, bidirectional=True):
        super(EfficientNet_LSTM, self).__init__()

        weights = EfficientNet_B0_Weights.DEFAULT
        self.efficientnet = efficientnet_b0(weights=weights)

        for param in self.efficientnet.parameters():
            param.requires_grad = False

        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

        lstm_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.efficientnet(x)
        features = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features)
        lstm_out = lstm_out[:, -1, :]
        output = self.classifier(lstm_out)
        return output

# --- Global Variables ---
app = Flask(__name__)
model = None

# --- Image Transforms (MATCHING YOUR main.py) ---
image_transforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((224, 224)), # Commented out
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Commented out
])

# --- Helper Functions ---
def allowed_file(filename):
    # --- allowed_file function remains the same ---
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model(weights_path):
    # --- load_model function remains the same ---
    global model
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}.")
    print(f"Loading model from {weights_path} onto {DEVICE}...")
    model = EfficientNet_LSTM().to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")

def generate_heatmap(input_pil_image, input_tensor_for_model, prob):
    # --- generate_heatmap function remains the same ---
    target_layer_name = 'efficientnet.features.8'
    heatmap_filename = f"heatmap_{uuid.uuid4()}.png"
    heatmap_save_path = os.path.join(RESULTS_FOLDER, heatmap_filename)
    heatmap_url = f"/results/{heatmap_filename}"

    model.eval()
    input_tensor_grad = input_tensor_for_model.clone().detach().requires_grad_(True).to(DEVICE)

    try:
        with GradCAM(model, target_layer=target_layer_name) as cam_extractor:
            scores = model(input_tensor_grad)
            cams = cam_extractor(class_idx=0, scores=scores)

            if not cams or len(cams) == 0:
                print("Grad-CAM returned no output.")
                return None

            heatmap_tensor = cams[0].squeeze(0).cpu()

            if not isinstance(heatmap_tensor, torch.Tensor) or heatmap_tensor.nelement() == 0:
                 print("Grad-CAM output tensor is invalid or empty.")
                 return None

            heatmap_pil = to_pil_image(heatmap_tensor, mode='F')
            heatmap_resized_pil = heatmap_pil.resize(input_pil_image.size, Image.Resampling.LANCZOS)
            result_pil = overlay_mask(input_pil_image, heatmap_resized_pil, alpha=0.5)

            os.makedirs(RESULTS_FOLDER, exist_ok=True)
            result_pil.save(heatmap_save_path)
            print(f"Heatmap saved to {heatmap_save_path}")
            return heatmap_url

    except ValueError as e:
        print(f"Error initializing/using Grad-CAM: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during Grad-CAM: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    # --- CORRECTED HTML/CSS (NO INVALID COMMENTS) ---
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deepfake Detector AI</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        html, body { height: 100%; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #e2e8f0;
            padding: 1.5rem;
            box-sizing: border-box;
        }
        .file-input-wrapper {
            position: relative;
            display: inline-block;
            overflow: hidden;
            cursor: pointer;
        }
        .file-input-button {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 0;
            cursor: pointer;
            z-index: 10;
        }
       .file-input-wrapper::before {
            content: 'Select Image';
            display: inline-block;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
            border-radius: 0.5rem;
            padding: 0.75rem 1.5rem;
            outline: none;
            white-space: nowrap;
            cursor: pointer;
            font-weight: 600;
            text-shadow: 0 1px 2px rgba(0,0,0,0.2);
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        .file-input-wrapper:hover::before {
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            transform: translateY(-2px);
        }
        .file-input-wrapper:active::before {
            background: linear-gradient(135deg, #4338ca, #6d28d9);
            transform: translateY(0px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        .detect-button {
            background: linear-gradient(135deg, #10b981, #34d399);
            color: white;
            font-weight: bold;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border: none;
        }
        .detect-button:hover {
            background: linear-gradient(135deg, #059669, #10b981);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            transform: translateY(-2px);
        }
        .detect-button:active {
            background: linear-gradient(135deg, #047857, #059669);
            transform: translateY(0px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        .detect-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            background: #9ca3af;
            box-shadow: none;
            transform: none;
        }
        .loader { border: 4px solid #e5e7eb; border-top: 4px solid #6366f1; border-radius: 50%; width: 32px; height: 32px; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .result-fake { color: #ef4444; }
        .result-real { color: #22c55e; }
        #heatmapLink img { transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out; }
        #heatmapLink:hover img { transform: scale(1.05); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        .card { border: 1px solid #e5e7eb; }
    </style>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
</head>
<body class="bg-slate-200">
    <div class="card bg-white rounded-xl shadow-2xl p-8 md:p-12 w-full max-w-lg mx-auto">
        <h1 class="text-3xl md:text-4xl font-bold text-gray-900 mb-8 text-center tracking-tight">Deepfake Detector</h1>
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="mb-8 text-center">
                <label for="imageUpload" class="block text-base font-medium text-gray-700 mb-4">Upload an image to analyze:</label>
                <div class="file-input-wrapper">
                    <input type="file" name="file" id="imageUpload" accept="image/png, image/jpeg" class="file-input-button"/>
                </div>
                <p id="fileNameDisplay" class="text-gray-600 text-sm mt-3 h-5"></p>
                <p id="fileError" class="text-red-600 text-sm mt-1 h-5 font-semibold"></p>
            </div>
            <div id="imagePreviewContainer" class="mb-8 hidden flex justify-center bg-gray-100 p-4 rounded-lg border border-gray-300">
                <img id="imagePreview" src="#" alt="Image Preview" class="max-w-full max-h-64 rounded-md object-contain shadow-inner"/>
            </div>
            <div class="text-center mb-8">
                <button type="submit" id="detectButton" class="detect-button">
                    Detect Deepfake
                </button>
            </div>
        </form>
        <div id="loadingIndicator" class="hidden flex justify-center items-center mb-6 py-4">
            <div class="loader"></div>
            <p class="ml-4 text-lg text-indigo-700 font-medium">Analyzing image...</p>
        </div>
        <div id="resultsContainer" class="hidden bg-slate-50 p-6 rounded-lg border border-slate-200 shadow-inner">
            <h2 class="text-xl font-semibold text-gray-800 mb-6 text-center">Analysis Results</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                <div class="text-center md:text-left space-y-2">
                    <div>
                        <p class="text-sm text-gray-500 font-medium uppercase tracking-wider">Prediction:</p>
                        <p id="predictionText" class="text-3xl font-bold"></p>
                    </div>
                    <div>
                        <p class="text-sm text-gray-500 font-medium uppercase tracking-wider mt-3">Confidence (Fake):</p>
                        <p id="probabilityText" class="text-xl font-semibold text-gray-900"></p>
                    </div>
                </div>
                <div class="flex justify-center md:justify-end items-center">
                     <a href="#" id="heatmapLink" target="_blank" title="Click to view full heatmap">
                        <img id="heatmapImage" src="#" alt="Grad-CAM Heatmap" class="w-40 h-40 md:w-44 md:h-44 rounded-lg shadow-md border border-gray-300 object-cover" onerror="this.src='https://placehold.co/200x200/e2e8f0/64748b?text=Heatmap+Error';"/>
                     </a>
                </div>
            </div>
        </div>
        <div id="messageBox" class="hidden mt-6 p-4 rounded-lg text-base text-center font-medium shadow-md"></div>
    </div>
    <script>
        // --- JavaScript remains the same ---
        const uploadForm = document.getElementById('uploadForm');
        const imageUpload = document.getElementById('imageUpload');
        const imagePreviewContainer = document.getElementById('imagePreviewContainer');
        const imagePreview = document.getElementById('imagePreview');
        const detectButton = document.getElementById('detectButton');
        const loadingIndicator = document.getElementById('loadingIndicator');
        const resultsContainer = document.getElementById('resultsContainer');
        const predictionText = document.getElementById('predictionText');
        const probabilityText = document.getElementById('probabilityText');
        const heatmapLink = document.getElementById('heatmapLink');
        const heatmapImage = document.getElementById('heatmapImage');
        const fileError = document.getElementById('fileError');
        const fileNameDisplay = document.getElementById('fileNameDisplay');
        const messageBox = document.getElementById('messageBox');
        let currentFile = null;

        imageUpload.addEventListener('change', (event) => {
            const file = event.target.files[0];
            fileError.textContent = '';
            fileNameDisplay.textContent = '';
            resultsContainer.classList.add('hidden');
            messageBox.classList.add('hidden');
            currentFile = null;
            imagePreviewContainer.classList.add('hidden');

            if (file) {
                if (!file.type.startsWith('image/')) {
                    fileError.textContent = 'Please select an image file (PNG or JPG).';
                    imageUpload.value = '';
                    return;
                }
                if (file.size > 10 * 1024 * 1024) { // Limit file size (e.g., 10MB)
                    fileError.textContent = 'File is too large (max 10MB).';
                    imageUpload.value = '';
                    return;
                }
                currentFile = file;
                fileNameDisplay.textContent = `Selected: ${file.name}`;
                const reader = new FileReader();
                reader.onload = function(e) {
                    imagePreview.src = e.target.result;
                    imagePreviewContainer.classList.remove('hidden');
                }
                reader.readAsDataURL(file);
            }
        });

        uploadForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            if (!currentFile) {
                showMessage('Please select an image file first.', 'error');
                return;
            }

            resultsContainer.classList.add('hidden');
            messageBox.classList.add('hidden');
            loadingIndicator.classList.remove('hidden');
            detectButton.disabled = true;

            const formData = new FormData();
            formData.append('file', currentFile);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData,
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || `Prediction failed (Status: ${response.status})`);
                }

                displayResults(data);

            } catch (error) {
                console.error("Error during prediction:", error);
                showMessage(error.message || 'An error occurred during analysis.', 'error');
            } finally {
                loadingIndicator.classList.add('hidden');
                detectButton.disabled = false;
            }
        });

        function displayResults(data) {
            const probabilityPercent = data.probability * 100;
            predictionText.textContent = data.decision;
            probabilityText.textContent = `${probabilityPercent.toFixed(2)}%`;

            const heatmapSrc = data.heatmap ? `${data.heatmap}?t=${new Date().getTime()}` : 'https://placehold.co/200x200/e2e8f0/64748b?text=No+Heatmap';
            heatmapLink.href = heatmapSrc;
            heatmapImage.src = heatmapSrc;

            predictionText.classList.remove('result-fake', 'result-real');
            probabilityText.classList.remove('text-red-700', 'text-green-700', 'text-orange-600', 'text-yellow-600');

            if (data.decision === 'FAKE') {
                predictionText.classList.add('result-fake');
                 if (probabilityPercent > 75) {
                    probabilityText.classList.add('text-red-700');
                } else {
                    probabilityText.classList.add('text-orange-600');
                }
            } else { // REAL
                predictionText.classList.add('result-real');
                if (probabilityPercent < 25) {
                    probabilityText.classList.add('text-green-700');
                } else {
                     probabilityText.classList.add('text-yellow-600');
                }
            }
            resultsContainer.classList.remove('hidden');
        }

        function showMessage(message, type = 'info') {
            messageBox.textContent = message;
            messageBox.classList.remove('hidden', 'bg-red-100', 'text-red-800', 'bg-blue-100', 'text-blue-800');
            messageBox.classList.add('shadow-md');
            if (type === 'error') {
                messageBox.classList.add('bg-red-100', 'text-red-800');
            } else {
                messageBox.classList.add('bg-blue-100', 'text-blue-800');
            }
        }
    </script>
</body>
</html>
    """
    return render_template_string(html_content)


@app.route('/predict', methods=['POST'])
def predict():
    # --- Prediction logic remains the same ---
    if model is None: return jsonify({"error": "Model not loaded"}), 500
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        try:
            img_bytes = file.read()
            img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')

            input_tensor = image_transforms(img_pil) # Shape: [C, H, W]

            input_tensor_model = input_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(input_tensor_model)
                prob_tensor = torch.sigmoid(outputs).squeeze()
                prob = prob_tensor.item()

            decision = "FAKE" if prob > 0.5 else "REAL"

            heatmap_url = generate_heatmap(img_pil, input_tensor_model, prob)

            return jsonify({
                "decision": decision,
                "probability": prob,
                "heatmap": heatmap_url if heatmap_url else "N/A"
            })

        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": "Failed to process image."}), 500
    else:
        return jsonify({"error": "Invalid file type."}), 400

# --- /results route and main execution remain the same ---
@app.route('/results/<filename>')
def uploaded_file(filename):
    safe_path = os.path.join(RESULTS_FOLDER, filename)
    if not os.path.exists(safe_path):
        app.logger.error(f"Heatmap file not found: {safe_path}")
        return "File not found", 404
    return send_from_directory(os.path.abspath(RESULTS_FOLDER), filename)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    try:
        load_model(MODEL_WEIGHTS_PATH)
    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}")
        exit(1)
    except Exception as e:
        print(f"FATAL ERROR loading model: {e}")
        exit(1)
    app.run(debug=True, host='0.0.0.0', port=5000)

