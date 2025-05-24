import os
import cv2
import torch
from ultralytics import YOLO

# Caminhos relativos ao diretório do script (tcc/utils)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

model_path = os.path.join(base_dir, "runs", "detect", "train", "weights", "best.pt")
image_dir = os.path.join(base_dir, "dataset", "images", "val")
label_dir = os.path.join(base_dir, "dataset", "labels", "val")
output_dir = os.path.join(base_dir, "comparacao_predicoes")
os.makedirs(output_dir, exist_ok=True)

# Carrega modelo
model = YOLO(model_path)

# Processa cada imagem
for fname in os.listdir(image_dir):
    if not fname.endswith(".jpg"):
        continue

    image_path = os.path.join(image_dir, fname)
    label_path = os.path.join(label_dir, fname.replace(".jpg", ".txt"))

    # Carrega imagem
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Desenha labels reais (vermelho)
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                _, x, y, bw, bh = map(float, line.strip().split())
                x1 = int((x - bw/2) * w)
                y1 = int((y - bh/2) * h)
                x2 = int((x + bw/2) * w)
                y2 = int((y + bh/2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Faz a inferência e desenha as predições (verde)
    results = model(image_path, conf=0.01, iou=0.3)
    for box in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, score, cls = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Salva imagem de comparação
    cv2.imwrite(os.path.join(output_dir, fname), img)

print(f"✅ Comparações salvas em: {output_dir}")
