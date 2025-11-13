# 🚗 Police Argos - Detecção de Placas Brasileiras

Sistema de detecção e reconhecimento automático de placas de veículos brasileiros usando YOLOv11 + EasyOCR.

## 📋 Funcionalidades

- ✅ Detecção de placas em **imagens** estáticas
- ✅ Detecção de placas em **vídeos**
- ✅ OCR local para leitura de placas (sem dependência de APIs)
- ✅ Suporte para placas antigas (ABC1234) e Mercosul (ABC1D23)
- ✅ Processamento otimizado para vídeos

## 🛠️ Tecnologias

- **YOLOv11** - Detecção de objetos
- **EasyOCR** - Reconhecimento óptico de caracteres
- **OpenCV** - Processamento de imagem/vídeo
- **Python 3.11+**

## 📦 Instalação

```bash
# Clone o repositório
git clone <seu-repositorio>
cd police-argos2

# Instale as dependências
cd backend
pip install -r requirements.txt
```

## 🚀 Como usar

### Para processar imagens:

1. Coloque sua imagem na pasta `backend/`
2. Edite o arquivo `detector.py` e defina:
```python
INPUT_PATH = "sua_imagem.jpg"
```
3. Execute:
```bash
python detector.py
```

### Para processar vídeos:

1. Coloque seu vídeo na pasta `backend/`
2. Edite o arquivo `detector.py` e defina:
```python
INPUT_PATH = "seu_video.mp4"
```
3. Execute:
```bash
python detector.py
```

## 📁 Estrutura do Projeto

```
police-argos2/
├── backend/
│   ├── detector.py              # Script principal
│   ├── requirements.txt         # Dependências Python
│   ├── placa-br-yolov11.pt     # Modelo YOLO treinado
│   └── [imagens/vídeos de teste]
└── README.md
```

## ⚙️ Configurações

No arquivo `detector.py`:

- `MODEL_PATH` - Caminho do modelo YOLO
- `INPUT_PATH` - Arquivo de entrada (imagem ou vídeo)
- `SKIP_FRAMES` - Processar 1 frame a cada N (para otimizar vídeos)

## 📊 Resultados

- **Imagens**: Salvas como `resultado_final.jpg`
- **Vídeos**: Salvos como `resultado_video.mp4`
- **Recortes**: Salvos como `placa_0.jpg`, `placa_1.jpg`, etc.

## 🔧 Requisitos do Sistema

- Python 3.11+
- 4GB+ RAM
- GPU (opcional, mas recomendado para vídeos)

## 📝 Licença

Este projeto é de código aberto.

## 👨‍💻 Autor

Felipe - Police Argos Project
