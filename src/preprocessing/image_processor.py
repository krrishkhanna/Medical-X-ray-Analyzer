"""
Image preprocessing module for X-ray analysis
"""

import cv2
import numpy as np
from skimage.feature import hog
from scipy import ndimage
import albumentations as A
from ..config import IMAGE_CONFIG, FEATURE_CONFIG, AUGMENTATION_CONFIG

class XRayPreprocessor:
    def __init__(self):
        self.target_size = IMAGE_CONFIG['TARGET_SIZE']
        self.augmentation = self._create_augmentation_pipeline()

    def _create_augmentation_pipeline(self):
        """Create an augmentation pipeline using albumentations"""
        return A.Compose([
            A.Rotate(limit=AUGMENTATION_CONFIG['ROTATION_RANGE']),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
        ])

    def preprocess_image(self, image, augment=False):
        """
        Main preprocessing pipeline for X-ray images
        
        Args:
            image: Input image (numpy array)
            augment: Whether to apply augmentation
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Resize image
        image = cv2.resize(image, self.target_size)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=IMAGE_CONFIG['CLAHE_CLIP_LIMIT'],
            tileGridSize=IMAGE_CONFIG['CLAHE_TILE_GRID_SIZE']
        )
        image = clahe.apply(image)
        
        # Apply Gaussian Blur
        image = cv2.GaussianBlur(
            image,
            IMAGE_CONFIG['GAUSSIAN_KERNEL_SIZE'],
            IMAGE_CONFIG['GAUSSIAN_SIGMA']
        )
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # Apply augmentation if requested
        if augment:
            augmented = self.augmentation(image=image)
            image = augmented['image']
            
        return image

    def extract_features(self, image):
        """
        Extract features from preprocessed image
        
        Args:
            image: Preprocessed image
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Extract HOG features
        features['hog'] = hog(
            image,
            orientations=FEATURE_CONFIG['HOG_ORIENTATIONS'],
            pixels_per_cell=FEATURE_CONFIG['HOG_PIXELS_PER_CELL'],
            cells_per_block=FEATURE_CONFIG['HOG_CELLS_PER_BLOCK'],
            visualize=False
        )
        
        # Apply Gabor filters
        gabor_responses = []
        for ksize in FEATURE_CONFIG['GABOR_KERNEL_SIZES']:
            for theta in FEATURE_CONFIG['GABOR_ORIENTATIONS']:
                kernel = cv2.getGaborKernel(
                    (ksize, ksize), sigma=4.0, theta=theta,
                    lambd=10.0, gamma=0.5, psi=0
                )
                gabor_response = cv2.filter2D(image, cv2.CV_32F, kernel)
                gabor_responses.append(gabor_response)
        
        features['gabor'] = np.array(gabor_responses)
        
        return features

    def detect_edges(self, image):
        """
        Detect edges in the image using Canny edge detector
        
        Args:
            image: Preprocessed image
            
        Returns:
            Edge map
        """
        # Auto-calculate thresholds
        sigma = 0.33
        median = np.median(image)
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        
        edges = cv2.Canny(
            (image * 255).astype(np.uint8),
            lower,
            upper
        )
        return edges 