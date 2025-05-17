import cv2
import os
import numpy as np

img_dir = "dataset/labels"  # imagens com círculos marcados
output_dir = "annotations_txt"
os.makedirs(output_dir, exist_ok=True)

# Azul específico: #4247C2 em BGR
target_bgr = np.array([194, 71, 66])
tolerance = 10

lower_bound = np.clip(target_bgr - tolerance, 0, 255)
upper_bound = np.clip(target_bgr + tolerance, 0, 255)

# Tamanho da bounding box centrada (ajustável)
box_size = 24  # pixels

for fname in os.listdir(img_dir):
    if not fname.endswith(".jpg"):
        continue

    img_path = os.path.join(img_dir, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Erro ao carregar imagem: {fname}")
        continue

    mask = cv2.inRange(img, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = img.shape[:2]
    label_lines = []

    for cnt in contours:
        # Encontra o centro do círculo desenhado
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        cx, cy = int(cx), int(cy)

        # Gera uma caixa pequena em torno do centro
        x1 = max(cx - box_size // 2, 0)
        y1 = max(cy - box_size // 2, 0)
        x2 = min(cx + box_size // 2, w - 1)
        y2 = min(cy + box_size // 2, h - 1)

        # Normaliza para formato YOLO
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h

        label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Salva anotações
    txt_path = os.path.join(output_dir, fname.replace(".jpg", ".txt"))
    with open(txt_path, "w") as f:
        f.write("\n".join(label_lines))

    print(f"{fname}: {len(label_lines)} eosinófilos anotados")
