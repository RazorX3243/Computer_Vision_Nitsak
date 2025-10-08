import cv2
import numpy as np
# rgb = bgr
# img[:] = 142, 142, 255
im = cv2.imread('portret.jpg')
img = cv2.resize(im, (640, 856))
cv2.rectangle(img,(40,180),(500,760),(142, 142, 255),2)
cv2.putText(img, "Super slay 67 Bohdasha Nitsak", (230, 780), cv2.FONT_HERSHEY_PLAIN, 1, (142, 142, 255), 2)



cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
