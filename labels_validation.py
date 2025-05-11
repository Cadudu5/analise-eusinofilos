import cv2
import os

image_path = "images/train/alguma_imagem.jpg"
label_path = "anotations_txt/imagem 1.txt"

img = cv2.imread(image_path)
h, w = img.shape[:2]

with open(label_path, 'r') as f:
    lines = f.readlines()

for line in lines:
    cls, x_center, y_center, box_w, box_h = map(float, line.split())
    x1 = int((x_center - box_w / 2) * w)
    y1 = int((y_center - box_h / 2) * h)
    x2 = int((x_center + box_w / 2) * w)
    y2 = int((y_center + box_h / 2) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Verificação das Anotações", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
