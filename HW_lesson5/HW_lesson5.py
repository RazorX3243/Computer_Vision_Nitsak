import cv2
import numpy as np
from numpy.ma.testutils import approx

img = cv2.imread('fig.jpg')
scale = 1
img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale))
print(img.shape)

img_copy = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([0, 28, 0])
upper = np.array([179, 255, 255])
mask = cv2.inRange(img, lower, upper)
img = cv2.bitwise_and(img, img, mask=mask)
img = cv2.GaussianBlur(img, (7, 7), 5)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 150:
        perimeter = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = round(w / h, 2)
        compactness = round((4 * np.pi * area) / (perimeter ** 2), 2)

        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            shape = "square"
        elif len(approx) > 8:
            shape = "star"
        else:
            shape = "circle"

        cv2.drawContours(img_copy, [cnt], -1, (255, 255, 255), 2)
        cv2.circle(img_copy, (cX, cY), 4, (0, 0, 255), -1)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)

        text_y = y + h + 15
        cv2.putText(img_copy, f'Shape: {shape}', (x, text_y),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        cv2.putText(img_copy, f'Area: {int(area)} px', (x, text_y + 15),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        cv2.putText(img_copy, f'Perimeter: {int(perimeter)} px', (x, text_y + 30),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        cv2.putText(img_copy, f'Compatness: {compactness}', (x, text_y + 45),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        cv2.putText(img_copy, f'Aspect ratio: {aspect_ratio}', (x, text_y + 60),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        cv2.putText(img_copy, f'Center: {cX}, {cY}', (x, text_y + 75),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)


cv2.imshow('img', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()