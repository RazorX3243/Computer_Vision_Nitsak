import cv2
import yt_dlp
from ultralytics import YOLO
import time
import pandas as pd
import os


YOUTUBE_URL = "https://www.youtube.com/watch?v=Lxqcg1qt0XU"

LINE_A = ((800, 400), (1600, 520))
LINE_B = ((400, 600), (1400, 800))

REAL_DISTANCE_METERS = 20


MIN_TIME = 0.3
MAX_TIME = 10


ydl_opts = {'format': 'best', 'quiet': True}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(YOUTUBE_URL, download=False)
    stream_url = info['url']

cap = cv2.VideoCapture(stream_url)


model = YOLO("yolov8n.pt")
CONF_THRESH = 0.4
TRACKER = "bytetrack.yaml"


def point_side(line, point):
    (x1, y1), (x2, y2) = line
    px, py = point
    return (x2 - x1)*(py - y1) - (y2 - y1)*(px - x1)

def crossed_line(prev_point, curr_point, line):
    side1 = point_side(line, prev_point)
    side2 = point_side(line, curr_point)
    return side1 * side2 < 0


previous_centers = {}
cross_times_A = {}
cross_times_B = {}
completed_ids = set()
object_speeds = {}
csv_data = []


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame,
                          conf=CONF_THRESH,
                          tracker=TRACKER,
                          persist=True,
                          verbose=False)

    r = results[0]

    # Малюємо лінії
    cv2.line(frame, LINE_A[0], LINE_A[1], (0, 0, 255), 3)
    cv2.line(frame, LINE_B[0], LINE_B[1], (255, 0, 0), 3)

    if r.boxes is not None and r.boxes.id is not None:

        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        ids = boxes.id.cpu().numpy()

        for i in range(len(xyxy)):

            class_id = int(cls[i])
            class_name = model.names[class_id]

            if class_name not in ["car", "truck", "bus"]:
                continue

            x1, y1, x2, y2 = xyxy[i].astype(int)
            tid = int(ids[i])

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            curr_point = (cx, cy)

            current_time = time.time()

            if tid not in previous_centers:
                previous_centers[tid] = curr_point

            prev_point = previous_centers[tid]

            # Перевірка перетину
            crossed_A = crossed_line(prev_point, curr_point, LINE_A)
            crossed_B = crossed_line(prev_point, curr_point, LINE_B)

            if crossed_A:
                cross_times_A[tid] = current_time

            if crossed_B:
                cross_times_B[tid] = current_time

            # Якщо є обидва перетини і ще не рахували
            if (tid in cross_times_A and
                tid in cross_times_B and
                tid not in completed_ids):

                time_diff = abs(cross_times_B[tid] - cross_times_A[tid])

                if MIN_TIME < time_diff < MAX_TIME:

                    speed_m_s = REAL_DISTANCE_METERS / time_diff
                    speed_kmh = speed_m_s * 3.6

                    completed_ids.add(tid)
                    object_speeds[tid] = round(speed_kmh, 1)

                    csv_data.append({
                        "ID": tid,
                        "Class": class_name,
                        "Speed_km_h": round(speed_kmh, 2),
                        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })

                    print(f"ID {tid} | {speed_kmh:.1f} km/h")

            previous_centers[tid] = curr_point

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{class_name} ID:{tid}"

            if tid in object_speeds:
                label += f" {object_speeds[tid]} km/h"

            cv2.putText(frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2)

    cv2.imshow("Speed Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

if len(csv_data) > 0:
    os.makedirs("output", exist_ok=True)
    df = pd.DataFrame(csv_data)
    df.to_csv("output/vehicles_speed.csv", index=False)
    print("CSV saved.")
else:
    print("No data saved.")
