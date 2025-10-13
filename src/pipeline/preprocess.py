import cv2
from torchvision import transforms
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

class FaceDetector:
    def __init__(self):
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 5)
        return faces
    
    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % 10 == 0:
                faces = self.detect(frame)
                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]
                    frames.append(face)
            frame_id += 1
        cap.release()
        return frames

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def save_faces_from_video(video_path, output_dir, detector, transform):
    faces = detector.extract_frames(video_path)
    basename = os.path.splitext(os.path.basename(video_path))[0]
    for i, face in enumerate(faces):
        face_img = cv2.resize(face, (224, 224))
        save_path = os.path.join(output_dir, f"{basename}_face_{i}.jpg")
        cv2.imwrite(save_path, face_img)

def process_video(args_tuple):
    video_path, output_dir = args_tuple
    detector = FaceDetector()  # Each thread gets its own detector (thread-safe)
    save_faces_from_video(video_path, output_dir, detector, transform)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input directory with videos")
    parser.add_argument("--output", type=str, required=True, help="Output directory for faces")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    video_files = [
        os.path.join(args.input, fname)
        for fname in os.listdir(args.input)
        if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ][:5]  # Only take the first 5 videos for testing CHANGE THIS LATER

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        executor.map(process_video, [(vf, args.output) for vf in video_files])