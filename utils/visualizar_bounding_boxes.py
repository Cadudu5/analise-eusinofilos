import cv2
import os
import numpy as np

img_dir = "dataset/labels"
output_dir = "visualizacoes"
os.makedirs(output_dir, exist_ok=True)

# Azul dos círculos: #4247C2 → BGR
target_bgr = np.array([194, 71, 66])
tolerance = 5  # AJUSTE fino da sensibilidade

lower_bound = np.clip(target_bgr - tolerance, 0, 255)
upper_bound = np.clip(target_bgr + tolerance, 0, 255)

box_size = 24  # Tamanho da caixa em pixels

for fname in os.listdir(img_dir):
    if not fname.endswith(".jpg"):
        continue

    img_path = os.path.join(img_dir, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Erro ao carregar: {fname}")
        continue

    h, w = img.shape[:2]
    
    # Máscara binária
    mask = cv2.inRange(img, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"{fname}: círculos detectados = {len(contours)}")

    # Cópia da imagem original para desenhar caixas
    img_caixas = img.copy()

    for cnt in contours:
        (cx, cy), _ = cv2.minEnclosingCircle(cnt)
        cx, cy = int(cx), int(cy)

        x1 = max(cx - box_size // 2, 0)
        y1 = max(cy - box_size // 2, 0)
        x2 = min(cx + box_size // 2, w - 1)
        y2 = min(cy + box_size // 2, h - 1)

        cv2.rectangle(img_caixas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Redimensiona as imagens para visualização lado a lado
    img_resized = cv2.resize(img, (256, 256))
    mask_resized = cv2.cvtColor(cv2.resize(mask, (256, 256)), cv2.COLOR_GRAY2BGR)
    caixas_resized = cv2.resize(img_caixas, (256, 256))

    combined = np.hstack((img_resized, mask_resized, caixas_resized))
    out_path = os.path.join(output_dir, f"vis_{fname}")
    cv2.imwrite(out_path, combined)

print("✅ Imagens de visualização salvas na pasta 'visualizacoes'.")
