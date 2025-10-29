---

# **Deepfake Detection System**

This project implements a **deep learning system for detecting deepfake images and videos** using PyTorch.
It leverages a pre-trained **EfficientNet-LSTM** model for robust feature extraction and temporal analysis, and includes explainability through **Grad-CAM** visualizations.
A simple **Flask-based web interface** allows users to upload files, run deepfake detection, view prediction confidence, and download a generated report.

---

## **Key Features**

* **Deepfake Detection:** Classifies input as either “REAL” or “FAKE”.
* **Multi-Modal Input:** Supports both **images** and **videos**.
* **Explainability:** Generates Grad-CAM heatmaps showing which regions influenced the model’s decision.
* **High Performance:** Fine-tuned model achieves high recall for detecting fake samples.
* **Data Handling:** Supports imbalanced datasets through oversampling and augmentation.
* **Web Interface:** User-friendly Flask frontend for quick testing, prediction, and report generation.

---

## **Project Structure**

```
DEEPFAKE DETECTION SYSTEM
├── .venv/                        # Virtual environment
├── configs/                      # Configuration files
├── data/                         # Dataset (requires Git LFS)
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   └── val/
│       ├── real/
│       └── fake/
├── src/
│   ├── datasets/                 # Dataset loading logic
│   ├── explainability/           # Grad-CAM and visualization
│   ├── models/                   # Model architectures (CNN-LSTM, EfficientNet-LSTM)
│   ├── evaluate.py               # Model evaluation script
│   ├── evaluate_video.py         # Video evaluation script
│   ├── finetune.py               # Fine-tuning script for failed cases
│   ├── main.py                   # Image prediction with Grad-CAM
│   ├── predict.py                # General prediction helper
│   ├── preprocess_frames.py      # Frame extraction from video
│   ├── train.py                  # Image training
│   ├── train_video.py            # Video-based model training
├── static/                       # Static files for Flask app
├── pipeline/                     # Data and processing pipelines
├── reports/                      # Generated reports
├── utils/                        # Helper utilities
├── best_img_model.pth            # Trained image model
├── best_video_model.pth          # Trained video model
├── best_model_finetuned.pth      # Fine-tuned final model
├── deepfake_app.py               # Flask web application
├── requirements.txt              # Project dependencies
├── failures.json                 # Misclassified cases (auto-generated)
├── README.md                     # Project documentation
└── .gitignore                    # Ignored files for Git
```

---

## **Setup Instructions**

### 1. Prerequisites

* Python 3.9 or higher
* Git ([https://git-scm.com/downloads](https://git-scm.com/downloads))
* Git LFS ([https://git-lfs.com](https://git-lfs.com))

### 2. Clone the Repository

```bash
git clone https://<YOUR_USERNAME>:<YOUR_PAT>@github.com/ruta28/deepfake-detection-system.git
cd deepfake-detection-system
git lfs pull
```

*(Replace `<YOUR_USERNAME>` and `<YOUR_PAT>` with your GitHub credentials.)*

### 3. Set Up the Virtual Environment

```bash
python -m venv .venv
# Activate
# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## **Usage**

### **1. Train the Model**

Train the model using EfficientNet-LSTM and oversampling:

```bash
python -m src.train
```

This saves the trained model weights as **`best_model.pth`** in the project root.

---

### **2. Evaluate Model Performance**

Evaluate performance and generate failure cases:

```bash
python -m src.evaluate
```

Outputs:

* Classification report
* Confusion matrix
* `failures.json` (list of failed predictions)

---

### **3. Fine-Tune the Model (Optional)**

Fine-tune using the misclassified samples for improved accuracy:

```bash
python -m src.finetune
```

Saves **`best_model_finetuned.pth`**.

---

### **4. Predict a Single Image (Explainability)**

Run inference and generate Grad-CAM:

```bash
python -m src.main --frame data/val/fake/sample_image.jpg --weights best_model_finetuned.pth --out evidence_sample
```

Example output:

```json
{
  "decision": "FAKE",
  "probability": 0.9821,
  "heatmap": "evidence_sample/gradcam.png"
}
```

---

### **5. Run the Flask Web App**

Launch the web interface to upload an image or video:

```bash
python deepfake_app.py
```

Then open the app in your browser at:

```
http://127.0.0.1:5000
```

#### **App Features:**

* Upload image or video files.
* Click **Detect Deepfake** to run inference.
* Displays:

  * Prediction label (**REAL/FAKE**)
  * Confidence score (%)
  * Grad-CAM heatmap
  * Button to download generated report

---

## **Example Web Interface**

A simple interface appears when you run the Flask app:

```
Deepfake Detector
-----------------
[ Select File ]
[ Detect Deepfake ]

→ Displays prediction and download link for report
```

---

## **Explainability (Grad-CAM)**

Grad-CAM helps visualize which parts of the face or frame influenced the model’s decision.
Generated heatmaps (e.g., `gradcam.png`) highlight high-attention regions in **red/yellow**.

---

## **Future Enhancements**

* Integrate real-time video stream detection.
* Experiment with Vision Transformer backbones.
* Build a full-stack deployment with FastAPI + React.
* Incorporate advanced methods (e.g., frequency domain or GAN fingerprint analysis).

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---