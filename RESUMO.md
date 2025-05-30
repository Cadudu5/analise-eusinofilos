# Análise Automatizada de Eosinófilos em Pólipos Nasais

## Visão Geral
Este projeto desenvolve um sistema de inteligência artificial para detectar e contar automaticamente eosinófilos em imagens histológicas de pólipos nasais. O objetivo é auxiliar no diagnóstico de subtipos de rinossinusite crônica e orientar o uso de tratamentos imunobiológicos.

## Tecnologias Principais
- **Framework de IA**: YOLOv8 (You Only Look Once versão 8)
- **Linguagem**: Python
- **Bibliotecas Principais**:
  - ultralytics (≥8.0.0) - Implementação do YOLOv8
  - opencv-python - Processamento de imagens
  - Pillow - Manipulação de imagens
  - scikit-image - Processamento avançado de imagens
  - pandas - Manipulação de dados

## Estrutura do Projeto
```
.
├── dataset/                  # Dataset original
├── dataset_preprocessed/     # Dataset após pré-processamento
├── utils/                    # Utilitários do projeto
├── annotations_txt/         # Anotações em formato texto
├── preprocess_dataset.py    # Script de pré-processamento
├── teste_yolo_gpu.py       # Script de teste do modelo
└── requirements.txt        # Dependências do projeto
```

## Pipeline de Processamento

### 1. Pré-processamento de Imagens
O script `preprocess_dataset.py` implementa várias técnicas de pré-processamento:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Ajuste de gamma (γ=1.5)
- Filtro bilateral para redução de ruído

### 2. Detecção de Objetos
- Utiliza a arquitetura YOLOv8 para detecção de eosinófilos
- Suporte a processamento em GPU para maior eficiência
- Configurável através de parâmetros de confiança

## Modelos Disponíveis
O projeto inclui diferentes versões do modelo YOLOv8:
- yolov8m.pt (50MB) - Versão média
- yolov8s.pt (22MB) - Versão pequena
- yolov8n.pt (6.2MB) - Versão nano

## Benefícios do Sistema
1. Redução do tempo de análise
2. Maior consistência nos diagnósticos
3. Eliminação da variabilidade entre observadores
4. Melhor acessibilidade à análise de eosinófilos
5. Suporte à pesquisa clínica

## Como Executar
1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Para pré-processar o dataset:
   ```bash
   python preprocess_dataset.py
   ```
3. Para testar o modelo:
   ```bash
   python teste_yolo_gpu.py
   ```

## Notas Técnicas
- O sistema é otimizado para GPU, mas também funciona em CPU
- Inclui suporte a diferentes tamanhos de modelo para equilibrar precisão e velocidade
- Implementa técnicas avançadas de processamento de imagem para melhorar a qualidade da detecção 