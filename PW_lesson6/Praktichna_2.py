import cv2
import numpy as np

img = cv2.imread('toys.png')
scale = 1
img = cv2.resize(img, (img.shape[1] * scale // 2, img.shape[0] * scale // 2))
print(img.shape)
img_copy = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([0, 45, 0])
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
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = round(w / h, 2)
        compactness = round((4 * np.pi * area) / (perimeter ** 2), 2)

        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        if len(approx) < 8:
            shape = "vorona"
        else:
            shape = "circle"

        cv2.drawContours(img_copy, [cnt], -1, (255, 255, 255), 2)
        cv2.circle(img_copy, (cX, cY), 4, (0, 255, 0), -1)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)

        text_y = y + h + 15
        cv2.putText(img_copy, f'Shape: {shape}', (x-20, text_y - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(img_copy, f'Area: {int(area)} px', (x-20, text_y),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(img_copy, f'Perimeter: {int(perimeter)} px', (x-20, text_y + 10),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(img_copy, f'Cordinates: x:{x}, y:{y}', (x-20, text_y + 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        if y == 86:
            cv2.putText(img_copy, f'Color: Green', (x - 20, text_y + 30),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        elif y == 57:
            cv2.putText(img_copy, f'Color: Red', (x - 20, text_y + 30),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        elif y == 263:
            cv2.putText(img_copy, f'Color: Blue', (x - 20, text_y + 30),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        elif y == 201:
            cv2.putText(img_copy, f'Color: Yellow', (x - 20, text_y + 30),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
cv2.imshow('img', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('result.jpg', img_copy)