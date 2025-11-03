import cv2
import numpy as np

net = cv2.dnn.readNetFromCaffe('Data/MobileNet/mobilenet_deploy.prototxt', 'Data/MobileNet/mobilenet.caffemodel')
classes = []
with open('Data/MobileNet/synset.txt', 'r', encoding = 'utf-8') as f: #open and check file
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(' ', 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)
image = cv2.imread('Images/MobileNet/cat.jpg')
blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1.0 / 127.5, (224, 224), (127.5, 127.5, 127.5))
net.setInput(blob)
preds = net.forward()
index = np.argmax(preds[0])
name = classes[index] if index < len(classes) else "unknown"
conf = float(preds[0][index])
print("Class:", name)
print("Likelihood:", conf)

text = f"{name}: {int(conf)}%"
cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
