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

    tracker = CentroidTracker()
    gradcam = GradCAM(model, model.conv5)

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    def process_frame(base64_img):
        frame = decode_base64_to_image(base64_img)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4)

        rects = [(x, y, x+w, y+h) for (x, y, w, h) in faces]
        objects = tracker.update(rects)

        # Default probs (no face)
        prob_dict = {label: 0 for label in CLASS_LABELS}

        for objectID, (centroid, bbox) in objects.items():
            (sx, sy, ex, ey) = bbox
            face = frame[sy:ey, sx:ex]
            if face.size == 0:
                continue

            tensor = preprocess_face(face)

            with torch.no_grad():
                out = model(tensor)
                probs = torch.softmax(out, dim=1)[0].cpu().numpy()

            # Convert to dict for UI
            prob_dict = {CLASS_LABELS[i]: float(probs[i]) for i in range(len(CLASS_LABELS))}

            cls_id = int(np.argmax(probs))
            label = CLASS_LABELS[cls_id]
            conf = float(probs[cls_id])

            # Generate GradCAM heatmap
            heatmap = gradcam.generate(tensor, cls_id)
            heatmap_resized = cv2.resize(heatmap, (ex - sx, ey - sy))
            heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            
            # Overlay heatmap on face region
            alpha = 0.4
            face_region = frame[sy:ey, sx:ex]
            frame[sy:ey, sx:ex] = cv2.addWeighted(face_region, 1 - alpha, heatmap_color, alpha, 0)

            # Draw bounding box + label
            cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf*100:.1f}%",
                        (sx, sy - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

        # Encode output
        frame_b64 = encode_image_to_base64(frame)

        # Wrap in JSON
        import json
        return json.dumps({
            "frame": frame_b64,
            "probs": prob_dict
        })

    return process_frame
