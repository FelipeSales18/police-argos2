# 🚗 Police Argos - Brazilian License Plate Detection

Automatic detection and recognition system for Brazilian vehicle license plates using YOLOv11 + EasyOCR.

## 📋 Features

- ✅ License plate detection in **static images**
- ✅ License plate detection in **videos**
- ✅ Local OCR for plate reading (no API dependencies)
- ✅ Support for old (ABC1234) and Mercosul (ABC1D23) plates
- ✅ Optimized video processing

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
2. Edit the `detector.py` file and set:
```python
INPUT_PATH = "your_image.jpg"
```
3. Run:
```bash
python detector.py
```

### To process videos:

1. Place your video in the `backend/` folder
2. Edit the `detector.py` file and set:
```python
INPUT_PATH = "your_video.mp4"
```
3. Run:
```bash
python detector.py
```

## 📁 Project Structure

```
police-argos2/
├── backend/
│   ├── detector.py              # Main script
│   ├── requirements.txt         # Python dependencies
│   ├── placa-br-yolov11.pt     # YOLO model (download from Hugging Face)
│   └── [test images/videos]
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## ⚙️ Configuration

In the [`detector.py`](backend/detector.py) file:

- `MODEL_PATH` - Path to the YOLO model
- `INPUT_PATH` - Input file (image or video)
- `SKIP_FRAMES` - Process 1 frame every N frames (to optimize videos)

## 📊 Results

- **Images**: Saved as `resultado_final.jpg`
- **Videos**: Saved as `resultado_video.mp4`
- **Crops**: Saved as `placa_0.jpg`, `placa_1.jpg`, etc.

## 🔧 System Requirements

- Python 3.11+
- 4GB+ RAM
- GPU (optional, but recommended for videos)
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
