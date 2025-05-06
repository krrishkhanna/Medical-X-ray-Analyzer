"""
Unit tests for image preprocessor
"""

import unittest
import numpy as np
import cv2
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.image_processor import XRayPreprocessor

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = XRayPreprocessor()
        # Create a sample test image (100x100 grayscale)
        self.test_image = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)

    def test_preprocess_image(self):
        """Test the image preprocessing function"""
        # Process the image
        processed = self.preprocessor.preprocess_image(self.test_image)
        
        # Check output shape matches target size
        self.assertEqual(processed.shape[:2], self.preprocessor.target_size)
        
        # Check output type and range
        self.assertEqual(processed.dtype, np.float32)
        self.assertTrue(0 <= processed.min() <= processed.max() <= 1)

    def test_augmentation(self):
        """Test image augmentation"""
        # Process the image with augmentation
        augmented = self.preprocessor.preprocess_image(self.test_image, augment=True)
        
        # Augmented image should still be in the right format
        self.assertEqual(augmented.shape[:2], self.preprocessor.target_size)
        self.assertEqual(augmented.dtype, np.float32)
        self.assertTrue(0 <= augmented.min() <= augmented.max() <= 1)

    def test_edge_detection(self):
        """Test edge detection"""
        # Apply edge detection
        edges = self.preprocessor.detect_edges(self.test_image / 255.0)
        
        # Should be a binary image
        self.assertTrue(np.array_equal(np.unique(edges), np.array([0, 255])) or 
                        np.array_equal(np.unique(edges), np.array([0])) or
                        np.array_equal(np.unique(edges), np.array([255])))

    def test_feature_extraction(self):
        """Test feature extraction"""
        # Normalize image for feature extraction
        normalized = self.test_image.astype(np.float32) / 255.0
        
        # Extract features
        features = self.preprocessor.extract_features(normalized)
        
        # Should contain both HOG and Gabor features
        self.assertIn('hog', features)
        self.assertIn('gabor', features)
        
        # Check that features are not empty
        self.assertGreater(len(features['hog']), 0)
        self.assertGreater(features['gabor'].size, 0)

if __name__ == '__main__':
    unittest.main() 