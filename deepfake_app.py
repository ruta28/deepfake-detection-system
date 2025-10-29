import os
import io
import uuid
import warnings
from PIL import Image
from datetime import datetime
import tempfile # For temporarily saving video
import cv2 # OpenCV for video processing
import math

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image, to_tensor # Need to_tensor
# Ensure scikit-image is installed: pip install scikit-image
from skimage.transform import resize

from flask import Flask, request, jsonify, render_template_string, send_from_directory

# --- PDF Report Generation Imports ---
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER, TA_LEFT

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Using a non-default target layer")
warnings.filterwarnings("ignore", message="Unable to retrieve source code") # Common torchcam warning


# --- Configuration ---
MODEL_WEIGHTS_PATH = "best_model_finetuned.pth"
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
# Added video extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAMES_PER_SECOND_TO_PROCESS = 1 # Process 1 frame per second

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

# --- Image Transforms (MATCHING TRAINING - ONLY ToTensor) ---
image_transforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((224, 224), antialias=True), # Commented out as per training
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Commented out as per training
])
# --- Video Transforms (MATCHING TRAINING - ONLY ToTensor) ---
video_frame_transforms = transforms.Compose([
    transforms.ToTensor(), # Converts numpy HWC [0,255] to torch CHW [0.0, 1.0]
    # transforms.Resize((224, 224), antialias=True), # Commented out as per training
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Commented out as per training
])


# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}

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
                return None, None

            heatmap_tensor = cams[0].squeeze(0).cpu()

            if not isinstance(heatmap_tensor, torch.Tensor) or heatmap_tensor.nelement() == 0:
                print("Grad-CAM output tensor is invalid or empty.")
                return None, None

            heatmap_pil = to_pil_image(heatmap_tensor, mode='F')
            heatmap_resized_pil = heatmap_pil.resize(input_pil_image.size, Image.Resampling.LANCZOS)
            result_pil = overlay_mask(input_pil_image, heatmap_resized_pil, alpha=0.5)

            os.makedirs(RESULTS_FOLDER, exist_ok=True)
            result_pil.save(heatmap_save_path)
            print(f"Heatmap saved to {heatmap_save_path}")
            return heatmap_save_path, heatmap_url

    except ValueError as e:
        print(f"Error initializing/using Grad-CAM: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during Grad-CAM: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# --- UPDATED generate_pdf_report Function ---
def generate_pdf_report(original_file_path, report_filename, decision, **kwargs):
    """
    Generates a PDF report for either image or video analysis.

    Args:
        original_file_path (str): Path to the uploaded original file (image or temp video).
        report_filename (str): Desired filename for the PDF report.
        decision (str): The final prediction ('REAL' or 'FAKE').
        **kwargs: Additional arguments depending on file type.
            For images: heatmap_image_path (str), probability (float)
            For videos: frames_processed (int), fake_frames (int), fake_percentage (float), confidence (float)
    """
    report_save_path = os.path.join(RESULTS_FOLDER, report_filename)
    report_url = f"/results/{report_filename}"

    doc = SimpleDocTemplate(report_save_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title = "Deepfake Analysis Report"
    p = Paragraph(title, styles['h1'])
    p.alignment = TA_CENTER
    story.append(p)
    story.append(Spacer(1, 0.2*inch))

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    p = Paragraph(f"Analysis Time: {timestamp}", styles['Normal'])
    p.alignment = TA_CENTER
    story.append(p)
    story.append(Spacer(1, 0.1*inch))

    original_filename = os.path.basename(original_file_path) # Use the base name
    p = Paragraph(f"Analyzed File: {original_filename}", styles['Italic'])
    p.alignment = TA_CENTER
    story.append(p)
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph("Analysis Results:", styles['h2']))
    story.append(Spacer(1, 0.1*inch))

    decision_style = styles['h3']
    if decision == "FAKE":
        decision_style.textColor = 'red'
    else:
        decision_style.textColor = 'green'

    story.append(Paragraph(f"Prediction: <font color='{decision_style.textColor}'>{decision}</font>", styles['Normal']))

    # --- Add details based on file type ---
    if 'probability' in kwargs: # Image
        probability = kwargs['probability']
        story.append(Paragraph(f"Confidence (Fake): {probability*100:.2f}%", styles['Normal']))
    elif 'confidence' in kwargs: # Video
        confidence = kwargs['confidence']
        frames_processed = kwargs.get('frames_processed', 0)
        fake_frames = kwargs.get('fake_frames', 0)
        fake_percentage = kwargs.get('fake_percentage', 0.0)
        story.append(Paragraph(f"Overall Confidence: {confidence:.2f}%", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("Video Details:", styles['h3']))
        story.append(Paragraph(f"Frames Analyzed: {frames_processed}", styles['Normal']))
        story.append(Paragraph(f"Frames Classified as FAKE: {fake_frames} ({fake_percentage:.1f}%)", styles['Normal']))
    # --- End details ---

    story.append(Spacer(1, 0.3*inch))

    # --- Add Visual Evidence only for Images ---
    if 'heatmap_image_path' in kwargs:
        heatmap_image_path = kwargs['heatmap_image_path']
        story.append(Paragraph("Visual Evidence:", styles['h2']))
        story.append(Spacer(1, 0.1*inch))

        try:
            # Use original_file_path directly as it was saved for images
            img_orig = ReportLabImage(original_file_path)
            img_orig.drawHeight = 2.5*inch * img_orig.drawHeight / img_orig.drawWidth
            img_orig.drawWidth = 2.5*inch
            story.append(Paragraph("Original Image:", styles['Normal']))
            story.append(img_orig)
            story.append(Spacer(1, 0.2*inch))
        except Exception as e:
            print(f"Error adding original image to report: {e}")
            story.append(Paragraph("<i>Error loading original image.</i>", styles['Italic']))

        if heatmap_image_path and os.path.exists(heatmap_image_path):
            try:
                img_heatmap = ReportLabImage(heatmap_image_path)
                img_heatmap.drawHeight = 2.5*inch * img_heatmap.drawHeight / img_heatmap.drawWidth
                img_heatmap.drawWidth = 2.5*inch
                story.append(Paragraph("Grad-CAM Heatmap:", styles['Normal']))
                story.append(img_heatmap)
            except Exception as e:
                print(f"Error adding heatmap image to report: {e}")
                story.append(Paragraph("<i>Error loading heatmap image.</i>", styles['Italic']))
        else:
            story.append(Paragraph("<i>Heatmap could not be generated or found.</i>", styles['Italic']))
    # --- End Visual Evidence ---

    try:
        doc.build(story)
        print(f"Report saved to {report_save_path}")
        return report_url
    except Exception as e:
        print(f"Error building PDF report: {e}")
        return None


# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    # --- UPDATED HTML with SCROLL FIX ---
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deepfake Detector</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* --- SCROLLING FIX: Removed height: 100%, align-items; Added min-height, padding-top/bottom --- */
        html { } /* Removed height: 100% */
        body {
            font-family: 'Inter', sans-serif;
            display: flex;
            /* align-items: center; <-- Removed */
            justify-content: center;
            background-color: #e2e8f0;
            padding: 1.5rem; /* Original padding */
            padding-top: 2rem; /* Added top padding */
            padding-bottom: 2rem; /* Added bottom padding */
            box-sizing: border-box;
            min-height: 100vh; /* Ensure body takes at least full viewport height */
            /* Removed height: 100%; */
        }
        .file-input-wrapper { position: relative; display: inline-block; overflow: hidden; cursor: pointer; }
        .file-input-button { position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0; cursor: pointer; z-index: 10; }
       .file-input-wrapper::before {
            content: 'Select File'; display: inline-block; background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white; border-radius: 0.5rem; padding: 0.75rem 1.5rem; outline: none; white-space: nowrap;
            cursor: pointer; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.2);
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
       .file-input-wrapper:hover::before { background: linear-gradient(135deg, #4f46e5, #7c3aed); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05); transform: translateY(-2px); }
       .file-input-wrapper:active::before { background: linear-gradient(135deg, #4338ca, #6d28d9); transform: translateY(0px); box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); }
        .detect-button { background: linear-gradient(135deg, #10b981, #34d399); color: white; font-weight: bold; padding: 0.75rem 1.5rem; border-radius: 0.5rem; transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1); box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); border: none; }
        .detect-button:hover { background: linear-gradient(135deg, #059669, #10b981); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05); transform: translateY(-2px); }
        .detect-button:active { background: linear-gradient(135deg, #047857, #059669); transform: translateY(0px); box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); }
        .detect-button:disabled { opacity: 0.5; cursor: not-allowed; background: #9ca3af; box-shadow: none; transform: none; }
        .report-button { display: inline-block; margin-top: 1rem; background: #3b82f6; color: white; font-weight: 600; padding: 0.6rem 1.2rem; border-radius: 0.5rem; text-decoration: none; transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1); box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); }
        .report-button:hover { background: #2563eb; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05); transform: translateY(-2px); }
        .report-button:active { background: #1d4ed8; transform: translateY(0px); box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); }
        .loader { border: 4px solid #e5e7eb; border-top: 4px solid #6366f1; border-radius: 50%; width: 32px; height: 32px; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .result-fake { color: #ef4444; }
        .result-real { color: #22c55e; }
        #heatmapLink img { transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out; }
        #heatmapLink:hover img { transform: scale(1.05); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        .card { border: 1px solid #e5e7eb; }
        #videoPreview { max-width: 100%; max-height: 16rem; }
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
                <label for="fileUpload" class="block text-base font-medium text-gray-700 mb-4">Upload an Image or Video:</label>
                <div class="file-input-wrapper">
                    <input type="file" name="file" id="fileUpload" accept="image/png, image/jpeg, video/mp4, video/avi, video/quicktime" class="file-input-button"/>
                </div>
                <p id="fileNameDisplay" class="text-gray-600 text-sm mt-3 h-5"></p>
                <p id="fileError" class="text-red-600 text-sm mt-1 h-5 font-semibold"></p>
            </div>
            <div id="previewContainer" class="mb-8 hidden flex justify-center bg-gray-100 p-4 rounded-lg border border-gray-300">
                <img id="imagePreview" src="#" alt="Image Preview" class="hidden max-w-full max-h-64 rounded-md object-contain shadow-inner"/>
                <video id="videoPreview" controls class="hidden max-w-full max-h-64 rounded-md object-contain shadow-inner"></video>
            </div>
            <div class="text-center mb-8">
                <button type="submit" id="detectButton" class="detect-button">
                    Detect Deepfake
                </button>
            </div>
        </form>
        <div id="loadingIndicator" class="hidden flex justify-center items-center mb-6 py-4">
            <div class="loader"></div>
            <p id="loadingText" class="ml-4 text-lg text-indigo-700 font-medium">Analyzing...</p>
        </div>
        <div id="resultsContainer" class="hidden bg-slate-50 p-6 rounded-lg border border-slate-200 shadow-inner">
            <h2 class="text-xl font-semibold text-gray-800 mb-6 text-center">Analysis Results</h2>
            
            <!-- Image Specific Layout -->
            <div id="imageResultLayout" class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
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
            
            <!-- Video Specific Layout (Added Confidence) -->
            <div id="videoResultLayout" class="hidden text-center space-y-3">
                 <div>
                     <p class="text-sm text-gray-500 font-medium uppercase tracking-wider">Overall Prediction:</p>
                     <p id="videoPredictionText" class="text-3xl font-bold"></p>
                 </div>
                 <div>
                     <p class="text-sm text-gray-500 font-medium uppercase tracking-wider mt-3">Overall Confidence:</p>
                     <p id="videoConfidenceText" class="text-xl font-semibold text-gray-900"></p> <!-- NEW -->
                 </div>
                 <div>
                     <p class="text-sm text-gray-500 font-medium uppercase tracking-wider mt-3">Details:</p>
                     <p id="videoDetailsText" class="text-base text-gray-700"></p>
                 </div>
            </div>
            
            <!-- Report Button Container (Used by both) -->
            <div id="reportButtonContainer" class="text-center mt-6">
                <a href="#" id="reportLink" class="report-button hidden" download>Download PDF Report</a>
            </div>
        </div>
        <div id="messageBox" class="hidden mt-6 p-4 rounded-lg text-base text-center font-medium shadow-md"></div>
    </div>
    
    <!-- UPDATED JavaScript -->
    <script>
        const uploadForm = document.getElementById('uploadForm');
        const fileUpload = document.getElementById('fileUpload');
        const previewContainer = document.getElementById('previewContainer');
        const imagePreview = document.getElementById('imagePreview');
        const videoPreview = document.getElementById('videoPreview');
        const detectButton = document.getElementById('detectButton');
        const loadingIndicator = document.getElementById('loadingIndicator');
        const loadingText = document.getElementById('loadingText');
        const resultsContainer = document.getElementById('resultsContainer');
        const imageResultLayout = document.getElementById('imageResultLayout');
        const predictionText = document.getElementById('predictionText');
        const probabilityText = document.getElementById('probabilityText');
        const heatmapLink = document.getElementById('heatmapLink');
        const heatmapImage = document.getElementById('heatmapImage');
        const videoResultLayout = document.getElementById('videoResultLayout');
        const videoPredictionText = document.getElementById('videoPredictionText');
        const videoConfidenceText = document.getElementById('videoConfidenceText'); // NEW
        const videoDetailsText = document.getElementById('videoDetailsText');
        const fileError = document.getElementById('fileError');
        const fileNameDisplay = document.getElementById('fileNameDisplay');
        const messageBox = document.getElementById('messageBox');
        const reportButtonContainer = document.getElementById('reportButtonContainer');
        const reportLink = document.getElementById('reportLink');
        let currentFile = null;
        let isVideo = false;

        fileUpload.addEventListener('change', (event) => {
            const file = event.target.files[0];
            fileError.textContent = '';
            fileNameDisplay.textContent = '';
            resultsContainer.classList.add('hidden');
            reportLink.classList.add('hidden');
            messageBox.classList.add('hidden');
            currentFile = null;
            previewContainer.classList.add('hidden');
            imagePreview.classList.add('hidden');
            videoPreview.classList.add('hidden');
            isVideo = false;

            if (file) {
                if (file.type.startsWith('image/')) { isVideo = false; }
                else if (file.type.startsWith('video/')) { isVideo = true; }
                else {
                    fileError.textContent = 'Please select an image or video file.';
                    fileUpload.value = ''; return;
                }
                const maxSize = isVideo ? 100 * 1024 * 1024 : 10 * 1024 * 1024;
                const maxSizeText = isVideo ? '100MB' : '10MB';
                if (file.size > maxSize) {
                    fileError.textContent = `File is too large (max ${maxSizeText}).`;
                    fileUpload.value = ''; return;
                }
                currentFile = file;
                fileNameDisplay.textContent = `Selected: ${file.name}`;
                const reader = new FileReader();
                reader.onload = function(e) {
                    if (isVideo) {
                        videoPreview.src = e.target.result;
                        videoPreview.classList.remove('hidden');
                    } else {
                        imagePreview.src = e.target.result;
                        imagePreview.classList.remove('hidden');
                    }
                    previewContainer.classList.remove('hidden');
                }
                reader.readAsDataURL(file);
            }
        });

        uploadForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            if (!currentFile) {
                showMessage('Please select a file first.', 'error'); return;
            }
            resultsContainer.classList.add('hidden');
            reportLink.classList.add('hidden');
            messageBox.classList.add('hidden');
            loadingIndicator.classList.remove('hidden');
            loadingText.textContent = isVideo ? 'Analyzing video (this may take a while)...' : 'Analyzing image...';
            detectButton.disabled = true;
            const formData = new FormData();
            formData.append('file', currentFile);
            formData.append('filename', currentFile.name); // Send original filename
            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();
                if (!response.ok) { throw new Error(data.error || `Prediction failed (Status: ${response.status})`); }
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
            resultsContainer.classList.remove('hidden');
            reportButtonContainer.classList.remove('hidden'); // Show container for both

            if (data.is_video) {
                imageResultLayout.classList.add('hidden');
                videoResultLayout.classList.remove('hidden');

                videoPredictionText.textContent = data.decision;
                // --- NEW: Display Video Confidence ---
                videoConfidenceText.textContent = `${data.confidence.toFixed(2)}%`;
                // ---
                videoDetailsText.textContent = `Analyzed ${data.frames_processed} frames. Found ${data.fake_frames} fake frames (${data.fake_percentage.toFixed(1)}%).`;

                videoPredictionText.classList.remove('result-fake', 'result-real');
                videoConfidenceText.classList.remove('text-red-700', 'text-green-700'); // Basic color for confidence
                if (data.decision === 'FAKE') {
                    videoPredictionText.classList.add('result-fake');
                    videoConfidenceText.classList.add('text-red-700');
                } else {
                    videoPredictionText.classList.add('result-real');
                    videoConfidenceText.classList.add('text-green-700');
                }

                // --- NEW: Show report link for video ---
                 if (data.report_url) {
                    reportLink.href = data.report_url;
                    reportLink.classList.remove('hidden');
                } else {
                    reportLink.classList.add('hidden');
                }
                // ---

            } else { // Image
                videoResultLayout.classList.add('hidden');
                imageResultLayout.classList.remove('hidden');

                const probabilityPercent = data.probability * 100;
                predictionText.textContent = data.decision;
                probabilityText.textContent = `${probabilityPercent.toFixed(2)}%`;
                const heatmapSrc = data.heatmap_url && data.heatmap_url !== 'N/A'
                    ? `${data.heatmap_url}?t=${new Date().getTime()}`
                    : 'https://placehold.co/200x200/e2e8f0/64748b?text=No+Heatmap';
                heatmapLink.href = heatmapSrc;
                heatmapImage.src = heatmapSrc;

                predictionText.classList.remove('result-fake', 'result-real');
                probabilityText.classList.remove('text-red-700', 'text-green-700', 'text-orange-600', 'text-yellow-600');
                if (data.decision === 'FAKE') {
                    predictionText.classList.add('result-fake');
                    if (probabilityPercent > 75) { probabilityText.classList.add('text-red-700'); }
                    else { probabilityText.classList.add('text-orange-600'); }
                } else {
                    predictionText.classList.add('result-real');
                    if (probabilityPercent < 25) { probabilityText.classList.add('text-green-700'); }
                    else { probabilityText.classList.add('text-yellow-600'); }
                }

                if (data.report_url) {
                    reportLink.href = data.report_url;
                    reportLink.classList.remove('hidden');
                } else {
                    reportLink.classList.add('hidden');
                }
            }
        }

        function showMessage(message, type = 'info') {
            messageBox.textContent = message;
            messageBox.classList.remove('hidden', 'bg-red-100', 'text-red-800', 'bg-blue-100', 'text-blue-800');
            messageBox.classList.add('shadow-md');
            if (type === 'error') { messageBox.classList.add('bg-red-100', 'text-red-800'); }
            else { messageBox.classList.add('bg-blue-100', 'text-blue-800'); }
        }
    </script>
</body>
</html>
    """
    return render_template_string(html_content)


@app.route('/predict', methods=['POST'])
def predict():
    # --- UPDATED Prediction logic ---
    if model is None: return jsonify({"error": "Model not loaded"}), 500
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    original_filename = request.form.get('filename', file.filename) # Get original filename

    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type."}), 400

    is_video = is_video_file(file.filename)
    # Use unique filenames to avoid conflicts, use original filename for report
    unique_id = uuid.uuid4()
    base_filename = f"{os.path.splitext(original_filename)[0]}_{unique_id}"
    upload_save_path = None # Define variable for cleanup
    video_save_path = None # Define variable for cleanup

    try:
        if not is_video:
            # --- Image processing ---
            img_bytes = file.read()
            img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')

            upload_filename = f"upload_{base_filename}{os.path.splitext(original_filename)[1]}"
            upload_save_path = os.path.join(UPLOAD_FOLDER, upload_filename)
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            # Save the uploaded file for the report
            with open(upload_save_path, "wb") as f_up: f_up.write(img_bytes)

            input_tensor = image_transforms(img_pil)
            input_tensor_model = input_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(input_tensor_model)
                prob_tensor = torch.sigmoid(outputs).squeeze()
                prob = prob_tensor.item()

            decision = "FAKE" if prob > 0.5 else "REAL"
            heatmap_abs_path, heatmap_rel_url = generate_heatmap(img_pil, input_tensor_model, prob)

            report_filename = f"report_{base_filename}.pdf"
            report_url = generate_pdf_report(
                original_file_path=upload_save_path, # Pass the saved image path
                report_filename=report_filename,
                decision=decision,
                # Image specific kwargs
                heatmap_image_path=heatmap_abs_path,
                probability=prob
            )

            return jsonify({
                "is_video": False,
                "decision": decision,
                "probability": prob,
                "heatmap_url": heatmap_rel_url if heatmap_rel_url else "N/A",
                "report_url": report_url if report_url else None
            })

        else:
            # --- Video processing ---
            temp_dir = tempfile.gettempdir()
            # Use a unique temp filename, but keep original for report title
            video_temp_filename = f"video_{unique_id}{os.path.splitext(original_filename)[1]}"
            video_save_path = os.path.join(temp_dir, video_temp_filename)
            file.save(video_save_path)
            print(f"Temporarily saved video to {video_save_path}")

            cap = cv2.VideoCapture(video_save_path)
            if not cap.isOpened(): raise IOError(f"Cannot open video file: {original_filename}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = int(fps / FRAMES_PER_SECOND_TO_PROCESS) if fps > 0 and FRAMES_PER_SECOND_TO_PROCESS > 0 else 1
            if frame_interval <= 0: frame_interval = 1 # Ensure interval is at least 1

            print(f"Video Info: FPS={fps:.2f}, Frames={frame_count}, Sampling Interval={frame_interval}")

            frame_predictions = []
            frames_processed = 0
            current_frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret: break

                if current_frame_idx % frame_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # --- IMPORTANT: Convert numpy array to PIL Image before transform ---
                    frame_pil = Image.fromarray(frame_rgb)
                    input_tensor = video_frame_transforms(frame_pil) # Use video transforms
                    # --- END CHANGE ---
                    input_tensor_model = input_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        outputs = model(input_tensor_model)
                        prob_tensor = torch.sigmoid(outputs).squeeze()
                        prob = prob_tensor.item()
                        frame_predictions.append(prob)
                        frames_processed += 1

                current_frame_idx += 1

            cap.release()

            if frames_processed == 0:
                 return jsonify({"error": "Could not process any frames from the video."}), 500

            fake_frames = sum(1 for p in frame_predictions if p > 0.5)
            fake_percentage = (fake_frames / frames_processed) * 100 if frames_processed > 0 else 0
            overall_decision = "FAKE" if fake_percentage >= 50 else "REAL" # Changed to >= 50

            # --- NEW: Calculate overall confidence ---
            # If FAKE, confidence is % fake frames. If REAL, confidence is % real frames.
            overall_confidence = fake_percentage if overall_decision == "FAKE" else (100.0 - fake_percentage)
            # ---

            print(f"Video Analysis: Processed={frames_processed}, Fake={fake_frames} ({fake_percentage:.1f}%), Decision={overall_decision}, Confidence={overall_confidence:.2f}%")

            # --- NEW: Generate PDF Report for Video ---
            report_filename = f"report_{base_filename}.pdf"
            report_url = generate_pdf_report(
                original_file_path=original_filename, # Pass original filename for title
                report_filename=report_filename,
                decision=overall_decision,
                # Video specific kwargs
                frames_processed=frames_processed,
                fake_frames=fake_frames,
                fake_percentage=fake_percentage,
                confidence=overall_confidence
            )
            # ---

            return jsonify({
                "is_video": True,
                "decision": overall_decision,
                "frames_processed": frames_processed,
                "fake_frames": fake_frames,
                "fake_percentage": fake_percentage,
                "confidence": overall_confidence, # NEW: Return confidence
                "heatmap_url": "N/A", # No heatmap for video
                "report_url": report_url if report_url else None # NEW: Return report URL
            })

    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Failed to process file."}), 500
    finally:
        # Cleanup temp files
        if upload_save_path and os.path.exists(upload_save_path):
             try: os.remove(upload_save_path)
             except Exception as e_clean: print(f"Error cleaning up image {upload_save_path}: {e_clean}")
        if video_save_path and os.path.exists(video_save_path):
             try: os.remove(video_save_path)
             except Exception as e_clean: print(f"Error cleaning up video {video_save_path}: {e_clean}")


# Route to serve generated results (heatmaps and reports)
@app.route('/results/<filename>')
def serve_result_file(filename):
    # --- This route remains the same ---
    # Basic security check
    if '..' in filename or filename.startswith('/'):
        return "Invalid filename", 400
    safe_path = os.path.abspath(os.path.join(RESULTS_FOLDER, filename))
    # Double-check it's still within the intended directory
    if not safe_path.startswith(os.path.abspath(RESULTS_FOLDER)):
         return "File access denied", 403
    if not os.path.exists(safe_path) or not filename.lower().endswith(('.png', '.pdf')):
        app.logger.error(f"Result file not found or invalid: {safe_path}")
        return "File not found", 404
    return send_from_directory(os.path.abspath(RESULTS_FOLDER), filename)

# --- Main Execution ---
if __name__ == '__main__':
    # --- Main execution remains the same ---
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