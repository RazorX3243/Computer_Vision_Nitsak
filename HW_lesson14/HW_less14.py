import os
import cv2
import shutil

PROJECT_DIR = os.path.dirname(__file__)

IMAGES_DIR = os.path.join(PROJECT_DIR, "images")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

OUT_DIR = os.path.join(PROJECT_DIR, "out")
PEOPLE_DIR = os.path.join(OUT_DIR, "people")
NO_PEOPLE_DIR = os.path.join(OUT_DIR, "no_people")

os.makedirs(PEOPLE_DIR, exist_ok=True)
os.makedirs(NO_PEOPLE_DIR, exist_ok=True)

PROTOTXT_PATH = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.prototxt")
MODEL_PATH = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.caffemodel")

net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

CLASSES = [
    "background",
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

PERSON_CLASS_ID = CLASSES.index("person")
CONF_THRESHOLD = 0.5


def detect_people(image_bgr):
    (h, w) = image_bgr.shape[:2]

    blob = cv2.dnn.blobFromImage(
        image_bgr,
        scalefactor=0.007843,
        size=(300, 300),
        mean=(127.5, 127.5, 127.5)
    )

    net.setInput(blob)
    detections = net.forward()

    boxes = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        class_id = int(detections[0, 0, i, 1])

        if class_id == PERSON_CLASS_ID and confidence >= CONF_THRESHOLD:
            box = detections[0, 0, i, 3:7]
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)
            boxes.append((x1, y1, x2, y2, confidence))
    return boxes


allowed_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
files = os.listdir(IMAGES_DIR)

for filename in files:
    if not filename.lower().endswith(allowed_ext):
        continue

    in_path = os.path.join(IMAGES_DIR, filename)
    img = cv2.imread(in_path)

    people_boxes = detect_people(img)
    people_count = len(people_boxes)

    boxed = img.copy()

    for (x1, y1, x2, y2, conf) in people_boxes:
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            boxed,
            f"person {conf:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.putText(
        boxed,f"People count: {people_count}",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0, 0, 255),2)

    if people_count > 0:
        out_path = os.path.join(PEOPLE_DIR, "boxed_" + filename)
        print(f"[PEOPLE] {filename}  count={people_count}")
    else:
        out_path = os.path.join(NO_PEOPLE_DIR, filename)
        print(f"[NO]     {filename}")

    cv2.imwrite(out_path, boxed)

print("\nГотово!")
print("Результати в:", OUT_DIR)
