# controllers/stream.py
import cv2
import torch
import numpy as np
import time
from models.loader import load_model
from services.tracker import CentroidTracker
from services.gradcam import GradCAM
import threading

# labels (ตามที่ให้มา)
CLASS_LABELS = ['happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

# face detector (OpenCV DNN face detector)
protoPath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# We'll attempt DNN model first (optional), but to keep dependency low use Haar as default.
face_cascade = cv2.CascadeClassifier(protoPath)

# load model
MODEL_PATH = "weights/model.pth"  # ปรับ path ตามต้องการ
model, device = load_model(MODEL_PATH)
# choose target layer for Grad-CAM (conv5)
target_layer = model.conv5
gradcam = GradCAM(model, target_layer)

tracker = CentroidTracker(maxDisappeared=40)

cap = None
cap_lock = threading.Lock()

def start_camera(src=0):
    global cap
    with cap_lock:
        if cap is None:
            cap = cv2.VideoCapture(src)
            # set resolution if needed
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def stop_camera():
    global cap
    with cap_lock:
        if cap is not None:
            cap.release()
            cap = None

def preprocess_face(face_bgr):
    # face_bgr: numpy BGR image (face crop)
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (64, 64))
    img = face_resized.astype(np.float32) / 255.0
    # normalize (you can change mean/std if needed)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # C,H,W
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    return tensor

def infer_and_annotate(frame):
    """
    frame: BGR image
    returns annotated frame (BGR)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    rects = []
    face_crops = []
    for (x, y, w, h) in faces:
        x1, y1, x2, y2 = x, y, x + w, y + h
        rects.append((x1, y1, x2, y2))
        face_crops.append(frame[y1:y2, x1:x2])

    objects = tracker.update(rects)

    # process each tracked object
    for objectID, (centroid, bbox) in objects.items():
        (startX, startY, endX, endY) = bbox
        # clamp to frame
        h, w = frame.shape[:2]
        sx = max(0, startX); sy = max(0, startY)
        ex = min(w - 1, endX); ey = min(h - 1, endY)
        if ex <= sx or ey <= sy:
            continue
        face = frame[sy:ey, sx:ex].copy()
        if face.size == 0:
            continue
        # preprocess
        tensor = preprocess_face(face)
        # inference
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
            label = CLASS_LABELS[pred_idx]

        # draw bbox like YOLO
        color = (0, 255, 0)
        cv2.rectangle(frame, (sx, sy), (ex, ey), color, 2)
        text = f"ID {objectID} {label}: {conf*100:.1f}%"
        # draw label background
        (tW, tH), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (sx, sy - tH - 6), (sx + tW, sy), color, -1)
        cv2.putText(frame, text, (sx, sy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        # Grad-CAM overlay: generate heatmap for predicted class
        try:
            heatmap = gradcam.generate(tensor, pred_idx)  # size 64x64 -> values 0-255
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap_color = cv2.resize(heatmap_color, (ex - sx, ey - sy))
            overlay = cv2.addWeighted(face, 0.5, heatmap_color, 0.5, 0)
            # place overlay back into frame but scaled to smaller inset (e.g., top-left corner of bbox)
            inset_h = int((ey - sy) * 0.35)
            inset_w = int((ex - sx) * 0.35)
            if inset_h > 10 and inset_w > 10:
                small = cv2.resize(overlay, (inset_w, inset_h))
                frame[sy:sy+inset_h, sx:sx+inset_w] = small
        except Exception as e:
            # if gradcam fails, ignore
            # print("GradCAM failed:", e)
            pass

    return frame

def generate_mjpeg():
    start_camera(0)
    global cap
    while True:
        with cap_lock:
            if cap is None:
                break
            ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        annotated = infer_and_annotate(frame)
        # encode
        ret2, jpeg = cv2.imencode('.jpg', annotated)
        if not ret2:
            continue
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
