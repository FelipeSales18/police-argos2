# 🚗 Police Argos - Brazilian License Plate Detection

Automatic detection and recognition system for Brazilian vehicle license plates using YOLOv11 + EasyOCR.

## 📋 Features

- ✅ License plate detection in **static images**
- ✅ Multi-scale detection for improved accuracy
- ✅ Local OCR for plate reading (no API dependencies)
- ✅ Voting system for best OCR results
- ✅ Confidence heatmap visualization
- ✅ Support for old (ABC1234) and Mercosul (ABC1D23) plates
- ✅ Advanced preprocessing (CLAHE, bilateral filter, Otsu thresholding)
- ✅ Debug mode with intermediate image outputs

## 🛠️ Technologies

- **YOLOv11** - Object detection
- **EasyOCR** - Optical character recognition
- **OpenCV** - Image/video processing
- **Python 3.11+**

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/FelipeSales18/police-argos2.git
cd police-argos2
```

### 2. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Download the trained YOLOv11 model

**⚠️ IMPORTANT**: You need to download the trained model for Brazilian license plate detection:

🔗 **Download**: [placa-br-yolov11.pt](https://huggingface.co/felipedutrain/placa-br-yolov11)

After downloading, place the `placa-br-yolov11.pt` file in the `backend/` folder.

## 🚀 How to Use

### To process images:

1. Place your image in the `backend/` folder
2. Edit the [`detector.py`](backend/detector.py) file and set:
```python
IMAGE_PATH = "your_image.jpg"
```
3. Run:
```bash
python detector.py
```

## 📁 Project Structure

```
police-argos2/
├── backend/
│   ├── detector.py              # Main detection script
│   ├── requirements.txt         # Python dependencies
│   ├── placa-br-yolov11.pt     # YOLO model (download from Hugging Face)
│   └── [test images]
├── LICENSE                      # MIT License
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## ⚙️ Configuration

In the [`detector.py`](backend/detector.py) file you can configure:

- `MODEL_PATH` - Path to the YOLO model (default: `"placa-br-yolov11.pt"`)
- `IMAGE_PATH` - Input image file (e.g., `"carro_teste.jpg"`)
- `DEBUG_MODE` - Enable/disable debug mode (saves intermediate images)
- `ENABLE_HEATMAP` - Enable/disable confidence heatmap generation

## 🔬 Advanced Features

### Multi-scale Detection
The system tests detection at 5 different scales (0.5x, 0.75x, 1.0x, 1.5x, 2.0x) to maximize accuracy.

### OCR Voting System
Tests 9 different image preprocessing techniques and uses a voting system to select the most reliable result:
- Grayscale
- Bilateral filter
- Otsu thresholding
- Inverted Otsu
- CLAHE enhancement
- Adaptive threshold
- Morphological operations
- And more...

### Confidence Heatmap
Generates visual heatmaps showing detection confidence across the image.

## 📊 Output Files

- `resultado_final.jpg` - Final image with bounding boxes and detected plates
- `placa_0.jpg`, `placa_1.jpg`, ... - Cropped license plate images
- `heatmap_overlay.jpg` - Confidence heatmap overlaid on original image
- `heatmap_puro.jpg` - Pure confidence heatmap
- `debug_*.jpg` - Debug images (when DEBUG_MODE is enabled)

## 🔧 System Requirements

- Python 3.11+
- 4GB+ RAM
- GPU (optional, but recommended)
- Windows, Linux or MacOS

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the project
2. Create a branch for your feature (`git checkout -b feature/MyFeature`)
3. Commit your changes (`git commit -m 'Add MyFeature'`)
4. Push to the branch (`git push origin feature/MyFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YOLOv11 trained model available on [Hugging Face](https://huggingface.co/felipedutrain/placa-br-yolov11)
- EasyOCR for the OCR library
- Ultralytics for the YOLO framework

## 👨‍💻 Author

**Felipe** - [GitHub](https://github.com/FelipeSales18)

---

⭐ If this project was useful to you, consider giving it a star!
