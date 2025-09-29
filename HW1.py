import cv2
img = cv2.imread("portret.jpg")
img2 = cv2.imread("gmail.jpg")
resized = cv2.resize(img, (500, 500))
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
resized2 = cv2.resize(img2, (500, 500))
gray2 = cv2.cvtColor(resized2, cv2.COLOR_BGR2GRAY)
edges2 = cv2.Canny(gray2, 100, 200)


cv2.imshow("image", edges)
cv2.imshow("image1", edges2)
cv2.imshow("image2", img2)
cv2.imshow("image3", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
