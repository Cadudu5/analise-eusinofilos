import cv2
import os

image_path = "dataset/images/imagem1_jpg.rf.8bcdacac23f0b1b24d24bcd7c4a0ce52.jpg"
label_path = "dataset/labels/imagem1_jpg.rf.8bcdacac23f0b1b24d24bcd7c4a0ce52.txt"  # Corrigido: 'anotations_txt' → 'annotations_txt'

# Verifica se a imagem existe
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

# Carrega a imagem
img = cv2.imread(image_path)
if img is None:
    raise ValueError(f"Erro ao carregar imagem: {image_path}")

# Verifica se o arquivo de anotações existe
if not os.path.exists(label_path):
    raise FileNotFoundError(f"Anotação não encontrada: {label_path}")

# Lê o tamanho da imagem
h, w = img.shape[:2]

# Lê e processa as anotações
with open(label_path, 'r') as f:
    lines = f.readlines()

for line in lines:
    cls, x_center, y_center, box_w, box_h = map(float, line.strip().split())
    x1 = int((x_center - box_w / 2) * w)
    y1 = int((y_center - box_h / 2) * h)
    x2 = int((x_center + box_w / 2) * w)
    y2 = int((y_center + box_h / 2) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Exibe a imagem anotada
cv2.imshow("Verificação das Anotações", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
