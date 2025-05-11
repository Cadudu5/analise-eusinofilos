import time
import torch
from ultralytics import YOLO

# Verifica se a GPU está disponível
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"Usando dispositivo: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

# Carrega o modelo
model = YOLO("yolov8n.pt")  # modelo pré-treinado

# Caminho da imagem de teste (coloque sua imagem aqui)
imagem_teste = "teste.jpg"

# Executa inferência e mede tempo
start = time.time()
results = model(imagem_teste, device=device, conf=0.3)
end = time.time()

# Exibe resultados
print(f"Tempo de inferência: {end - start:.3f} segundos")
print(f"Número de objetos detectados: {len(results[0].boxes)}")

# Mostra a imagem com os resultados (opcional)
results[0].show()
