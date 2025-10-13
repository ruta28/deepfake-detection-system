import cv2
import os
from mtcnn import MTCNN
from tqdm import tqdm

detector = MTCNN()

def extract_faces_from_video(video_path, output_dir, max_frames=50):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # sample every 10th frame
        if frame_count % 10 != 0:
            continue

        results = detector.detect_faces(frame)
        if results:
            x, y, w, h = results[0]['box']
            face = frame[y:y+h, x:x+w]
            face_path = os.path.join(output_dir, f"frame_{saved_count}.jpg")
            cv2.imwrite(face_path, face)
            saved_count += 1

    cap.release()

if __name__ == "__main__":
    input_video = "data/ffpp/original_sequences/youtube/c23/videos/000.mp4"
    output_dir = "data/processed/original/000"
    extract_faces_from_video(input_video, output_dir)
