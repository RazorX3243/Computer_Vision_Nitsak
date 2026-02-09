import os
import cv2
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
VIDEO_PATH = os.path.join(PROJECT_DIR, "video", "video.mp4")

CONF_THRESHOLD = 0.4
RESIZE_WIDTH = 960

TRANSPORT_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(VIDEO_PATH)
prev_time = time.time()
fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if RESIZE_WIDTH:
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))

    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

    transport_count = {name: 0 for name in TRANSPORT_CLASSES.values()}

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls not in TRANSPORT_CLASSES:
                continue

            label_name = TRANSPORT_CLASSES[cls]
            transport_count[label_name] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label_name} {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    now = time.time()
    fps = 1.0 / (now - prev_time)
    prev_time = now

    y_offset = 30
    for name, count in transport_count.items():
        cv2.putText(
            frame,
            f"{name}: {count}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )
        y_offset += 30

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Traffic detection", frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('й')]:
        break

cap.release()
cv2.destroyAllWindows()
