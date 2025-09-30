import cv2
import numpy as np
img = np.zeros((512,512,3), np.uint8)
# rgb = bgr
# img[:] = 142, 142, 255
img[100:200, 200:300] = 142, 142, 255


cv2.rectangle(img,(100,100),(200,200),(255,255,255),2)
cv2.line(img,(100,100),(200,200),(255,255,255),2)
cv2.line(img,(200,100),(100,200),(255,255,255),2)
print(img.shape)
cv2.line(img, (0, img.shape[0] // 2), (img.shape[1], img.shape[0] // 2), (255,255,255), 2)
cv2.putText(img, "Bagadan N", (200, 250), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 1)
cv2.circle(img,(200,200),200,(255,255,255),2)



cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()