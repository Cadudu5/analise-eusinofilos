import cv2
import os
import numpy as np

img_dir = "dataset/labels/"  # imagens com os círculos azuis
output_dir = "annotations_txt"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(img_dir):
    if not fname.endswith(".jpg"): continue

    img_path = os.path.join(img_dir, fname)
    img = cv2.imread(img_path)

    # converte para HSV para detectar azul
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # encontra contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = img.shape[:2]
    label_lines = []

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)

        # normaliza
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        width = bw / w
        height = bh / h

        label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # salva o .txt correspondente
    txt_path = os.path.join(output_dir, fname.replace(".jpg", ".txt"))
    with open(txt_path, "w") as f:
        f.write("\n".join(label_lines))
