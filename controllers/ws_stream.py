# controllers/ws_stream.py
import base64
import cv2
import numpy as np
import torch

from models.loader import load_model
from services.gradcam import GradCAM
from services.tracker import CentroidTracker

CLASS_LABELS = ['happiness','surprise','sadness','anger','disgust','fear']

# -----------------------------------------------------------
# Load model only once globally (shared weights)
# -----------------------------------------------------------
model, device = load_model("weights/model.pth")

# -----------------------------------------------------------
# Helpers for base64 encode/decode
# -----------------------------------------------------------
def decode_base64_to_image(base64_string):
    img_bytes = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame

def encode_image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode()

def preprocess_face(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (64, 64))
    face_resized = face_resized.astype(np.float32) / 255.0
    face_resized = np.transpose(face_resized, (2, 0, 1))
    return torch.tensor(face_resized).unsqueeze(0).to(device)

# ============================================================
#   FACTORY: Create new processing instance for each client
# ============================================================
def create_client_processor():
    """
    สำหรับ Client แต่ละราย → สร้าง tracker และ gradcam คนละตัว
    """
    tracker = CentroidTracker()
    gradcam = GradCAM(model, model.conv5)

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    def process_frame(base64_img):
        frame = decode_base64_to_image(base64_img)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4)

        rects = []
        for (x, y, w, h) in faces:
            rects.append((x, y, x + w, y + h))

        objects = tracker.update(rects)

        for objectID, (centroid, bbox) in objects.items():
            (sx, sy, ex, ey) = bbox
            face = frame[sy:ey, sx:ex]

            if face.size == 0:
                continue

            tensor = preprocess_face(face)

            with torch.no_grad():
                out = model(tensor)
                probs = torch.softmax(out, dim=1)[0].cpu().numpy()

            cls_id = int(np.argmax(probs))
            conf = float(probs[cls_id])
            label = CLASS_LABELS[cls_id]

            # Draw bbox
            cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf*100:.1f}%",
                        (sx, sy - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        return encode_image_to_base64(frame)

    return process_frame
