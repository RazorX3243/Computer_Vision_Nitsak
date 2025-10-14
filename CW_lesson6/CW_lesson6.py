import cv2
import numpy as np

cap = cv2.VideoCapture(0)
ret, frame1 = cap.read()
grey1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
grey1 = cv2.convertScaleAbs(grey1, alpha=1.5, beta=10)

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    grey2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    grey2 = cv2.convertScaleAbs(grey2, alpha=1.5, beta=5)

    diff = cv2.absdiff(grey1, grey2)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 800:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame = cv2.convertScaleAbs(frame, alpha=5, beta=255)

    cv2.imshow('frame', frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break






cap.release()
cv2.destroyAllWindows()