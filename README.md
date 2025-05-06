# Medical X-ray Analyzer

An AI-powered system for analyzing medical X-ray images to detect conditions like fractures and pneumonia. The system uses deep learning with MobileNetV2/ResNet18 architecture and provides visual explanations using Grad-CAM.

## Features

- Support for both MURA (bone) and ChestX-ray14 datasets
- Advanced image preprocessing pipeline with CLAHE and edge detection
- Feature extraction using HOG and Gabor filters
- Transfer learning with MobileNetV2/ResNet18
- Model interpretability with Grad-CAM visualization
- User-friendly Streamlit interface
- Efficient data loading and processing

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd x-ray-analyzer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Setup

1. Download the datasets:
   - MURA dataset: https://stanfordmlgroup.github.io/competitions/mura/
   - ChestX-ray14 dataset: https://nihcc.app.box.com/v/ChestXray-NIHCC

2. Place the datasets in the appropriate directories:
   - MURA: `data/raw/mura/`
   - ChestX-ray14: `data/raw/chestxray14/`

## Training

To train the model:

```bash
# Train on MURA dataset with MobileNetV2
python src/train.py --dataset mura --model-type mobilenetv2

# Train on ChestX-ray14 dataset with ResNet18
python src/train.py --dataset chestxray --model-type resnet18
```

## Running the UI

To start the Streamlit interface:

```bash
streamlit run src/ui/app.py
```

The UI will be available at `http://localhost:8501`.

## Project Structure

```
.
├── data/
│   ├── raw/             # Raw datasets
│   └── processed/       # Processed data
├── src/
│   ├── data/           # Data loading and processing
│   ├── preprocessing/  # Image preprocessing
│   ├── models/        # Model architecture and training
│   ├── visualization/ # Grad-CAM visualization
│   ├── ui/           # Streamlit interface
│   ├── config.py     # Configuration
│   └── train.py      # Training script
├── tests/            # Unit tests
├── requirements.txt  # Dependencies
└── README.md        # Documentation
```

## Model Performance

The system achieves the following performance metrics (example):
- Accuracy: ~85% on MURA dataset
- AUC-ROC: ~0.90
- Precision: ~0.83
- Recall: ~0.87

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MURA dataset: Stanford ML Group
- ChestX-ray14 dataset: NIH Clinical Center
- MobileNetV2 and ResNet architectures 