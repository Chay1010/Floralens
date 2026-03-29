# 🌿 FloraLens — AI-Powered Plant & Flower Scanner

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FloraLens is an end-to-end deep learning application that identifies plant species from photographs.  
It uses a fine-tuned **EfficientNet-B2** on the **Oxford 102 Flowers** dataset, served via a **FastAPI** backend with **ONNX Runtime** inference, and a modern responsive web frontend.

## 🏗️ Architecture

```
floralens/
├── backend/                # FastAPI server + ML pipeline
│   ├── app/                # Application code
│   │   ├── main.py         # FastAPI routes & startup
│   │   ├── model.py        # Model loading & inference
│   │   ├── preprocessing.py# Image preprocessing pipeline
│   │   └── config.py       # Configuration & constants
│   ├── training/           # Model training pipeline
│   │   ├── train.py        # Training loop with W&B logging
│   │   ├── dataset.py      # Oxford 102 Flowers dataloader
│   │   ├── evaluate.py     # Evaluation & error analysis
│   │   ├── export_onnx.py  # ONNX conversion
│   │   └── profile_model.py# Performance profiling
│   ├── tests/              # Unit & integration tests
│   │   ├── test_preprocessing.py
│   │   └── test_api.py
│   ├── experiments/        # Experiment tracking logs
│   ├── Dockerfile          # Production container
│   ├── requirements.txt    # Python dependencies
│   └── environment.yml     # Conda environment
├── frontend/               # Web UI
│   ├── index.html
│   ├── style.css
│   └── app.js
└── README.md
```

## 🚀 Quick Start

### Backend

```bash
cd backend
conda env create -f environment.yml
conda activate floralens
python -m training.train --lr 3e-4 --epochs 25 --batch-size 32
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
cd backend
docker build -t floralens:latest .
docker run -p 8000:8000 floralens:latest
```

### Frontend

Open `frontend/index.html` in your browser or serve with any static file server, pointed at `http://localhost:8000`.

## 📊 Results

| Model | Val Accuracy | Val Loss | Params | Inference (ms) |
|-------|-------------|----------|--------|----------------|
| EfficientNet-B2 (ours) | **89.7%** | 0.412 | 7.8M | 12.3 |
| ResNet-50 (baseline) | 85.2% | 0.583 | 23.5M | 18.7 |
| MobileNetV3 (lightweight) | 82.1% | 0.691 | 3.4M | 8.1 |

## 📝 License

MIT License — see [LICENSE](LICENSE).

## 👤 Author

Chay — chay101045@gmail.com
