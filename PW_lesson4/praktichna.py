import cv2
import numpy as np

img = np.zeros((400,600,3), np.uint8)
im = cv2.imread("portret.jpg")
qr = cv2.imread("qr_code.png")
res = cv2.resize(im, (150, 170))
res1 = cv2.resize(qr, (100, 100))
img[:] = 255, 219, 241
cv2.rectangle(img, (10,10), (img.shape[1]-10,img.shape[0]-10), (186, 161, 176), 3)
x, y = 50, 50
img[y:y+res.shape[0], x:x+res.shape[1]] = res
x1, y1 = img.shape[1]-res1.shape[1]-30, img.shape[0]-res1.shape[0]-30
img[y1:y1+res1.shape[0], x1:x1+res1.shape[1]] = res1

cv2.putText(img, "Bogomdan Nitsak", (240, 100), cv2.FONT_HERSHEY_PLAIN, 2, (186, 161, 176), 2)
cv2.putText(img, "OpenCV Business Card", (70, 350), cv2.FONT_HERSHEY_PLAIN, 2, (186, 161, 176), 2)
cv2.putText(img, "Computer Vision Student", (240, 130), cv2.FONT_HERSHEY_PLAIN, 1, (186, 161, 176), 2)
cv2.putText(img, "Email: bohdan.nitsak@gmail.com", (240, 160), cv2.FONT_HERSHEY_PLAIN, 1, (186, 161, 176), 1)
cv2.putText(img, "Photo: +380 68 833 2591", (240, 190), cv2.FONT_HERSHEY_PLAIN, 1, (186, 161, 176), 1)
cv2.putText(img, "Email: 05/01/2009", (240, 220), cv2.FONT_HERSHEY_PLAIN, 1, (186, 161, 176), 1)


print(img.shape)
cv2.imwrite("business_card.png", img)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
