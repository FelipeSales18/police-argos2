# 🚗 Police Argos - Detecção de Placas Brasileiras

Sistema de detecção e reconhecimento automático de placas de veículos brasileiros usando YOLOv11 + EasyOCR.

## 📋 Funcionalidades

- ✅ Detecção de placas em **imagens** estáticas
- ✅ OCR local para leitura de placas (sem dependência de APIs)
- ✅ Suporte para placas antigas (ABC1234) e Mercosul (ABC1D23)
- ✅ Processamento otimizado para vídeos

## 🛠️ Tecnologias

- **YOLOv11** - Detecção de objetos
- **EasyOCR** - Reconhecimento óptico de caracteres
- **OpenCV** - Processamento de imagem/vídeo
- **Python 3.11+**

## 📦 Instalação

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/police-argos2.git
cd police-argos2
```

### 2. Instale as dependências
```bash
cd backend
pip install -r requirements.txt
```

### 3. Baixe o modelo YOLOv11 treinado

**⚠️ IMPORTANTE**: Você precisa baixar o modelo treinado para detecção de placas brasileiras:

🔗 **Download**: [placa-br-yolov11.pt](https://huggingface.co/felipedutrain/placa-br-yolov11)

Após o download, coloque o arquivo `placa-br-yolov11.pt` na pasta `backend/`.

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
│   ├── placa-br-yolov11.pt     # Modelo YOLO (baixar do Hugging Face)
│   └── [imagens/vídeos de teste]
├── LICENSE                      # Licença MIT
└── README.md                    # Este arquivo
```

## ⚙️ Configurações

No arquivo [`detector.py`](backend/detector.py):

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
- Windows, Linux ou MacOS

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:

1. Fazer um Fork do projeto
2. Criar uma branch para sua feature (`git checkout -b feature/MinhaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona MinhaFeature'`)
4. Push para a branch (`git push origin feature/MinhaFeature`)
5. Abrir um Pull Request

## 📝 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- Modelo YOLOv11 treinado disponível em [Hugging Face](https://huggingface.co/felipedutrain/placa-br-yolov11)
- EasyOCR pela biblioteca de OCR
- Ultralytics pelo framework YOLO

## 👨‍💻 Autor

**Felipe** - [GitHub](https://github.com/seu-usuario)

---

⭐ Se este projeto foi útil para você, considere dar uma estrela!
