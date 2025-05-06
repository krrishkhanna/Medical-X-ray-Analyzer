"""
Configuration file for X-ray Analysis System
"""

# Data Configuration
DATA_CONFIG = {
    'MURA_DATASET_PATH': 'data/raw/mura',
    'CHESTXRAY_DATASET_PATH': 'data/raw/chestxray14',
    'PROCESSED_DATA_PATH': 'data/processed',
    'TRAIN_SPLIT': 0.8,
    'VAL_SPLIT': 0.1,
    'TEST_SPLIT': 0.1,
}

# Image Processing Configuration
IMAGE_CONFIG = {
    'TARGET_SIZE': (224, 224),
    'CHANNELS': 1,
    'CLAHE_CLIP_LIMIT': 2.0,
    'CLAHE_TILE_GRID_SIZE': (8, 8),
    'GAUSSIAN_KERNEL_SIZE': (5, 5),
    'GAUSSIAN_SIGMA': 1.0,
}

# Model Configuration
MODEL_CONFIG = {
    'MODEL_TYPE': 'mobilenetv2',  # or 'resnet18'
    'NUM_CLASSES': 2,
    'BATCH_SIZE': 32,
    'EPOCHS': 15,
    'LEARNING_RATE': 0.0001,
    'EARLY_STOPPING_PATIENCE': 5,
    'CHECKPOINT_PATH': 'models/checkpoints',
}

# Augmentation Configuration
AUGMENTATION_CONFIG = {
    'ROTATION_RANGE': 20,
    'WIDTH_SHIFT_RANGE': 0.2,
    'HEIGHT_SHIFT_RANGE': 0.2,
    'HORIZONTAL_FLIP': True,
    'ZOOM_RANGE': 0.2,
}

# Feature Extraction Configuration
FEATURE_CONFIG = {
    'HOG_ORIENTATIONS': 9,
    'HOG_PIXELS_PER_CELL': (8, 8),
    'HOG_CELLS_PER_BLOCK': (2, 2),
    'GABOR_KERNEL_SIZES': [5, 7, 9],
    'GABOR_ORIENTATIONS': [0, 45, 90, 135],
}

# UI Configuration
UI_CONFIG = {
    'UPLOAD_PATH': 'uploads',
    'MAX_FILE_SIZE': 5 * 1024 * 1024,  # 5MB
    'ALLOWED_EXTENSIONS': ['.jpg', '.jpeg', '.png', '.dicom'],
} 