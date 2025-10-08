import cv2
import numpy as np

img = cv2.imread('pazany.png')



img_copy = img.copy()
img_copy_color = img_copy.copy()
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
img_copy = cv2.GaussianBlur(img_copy, (5, 5), 2)
img_copy = cv2.Canny(img_copy, 150, 120)
contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 100:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.drawContours(img_copy_color, [cnt], -1, (0, 255, 0), 2)
        cv2.rectangle(img_copy_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text_y = y - 5 if y - 5 > 10 else y + 15
        text_x = f'x:{x} y:{y} S:{int(area)}'

        cv2.putText(img_copy_color, text_x, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imshow('Image', img)
cv2.imshow('Canny', img_copy)
cv2.imshow('Contours and Bounding Boxes', img_copy_color)

cv2.waitKey(0)
cv2.destroyAllWindows()

