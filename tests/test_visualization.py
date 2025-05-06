"""
Unit tests for GradCAM visualization
"""

import unittest
import numpy as np
import tensorflow as tf
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.xray_model import XRayModel
from src.visualization.grad_cam import GradCAM

class TestGradCAM(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.model = XRayModel()
        self.grad_cam = GradCAM(self.model.model)
        # Create a sample test image (with proper dimensions)
        self.test_image = np.random.rand(224, 224, 1).astype(np.float32)

    def test_grad_cam_initialization(self):
        """Test GradCAM initialization"""
        # Check that the GradCAM object was initialized
        self.assertIsNotNone(self.grad_cam)
        
        # Check that the layer name is set
        self.assertIsNotNone(self.grad_cam.layer_name)
        
        # Check that the grad model is created
        self.assertIsNotNone(self.grad_cam.grad_model)

    def test_compute_heatmap(self):
        """Test heatmap computation"""
        # Compute heatmap
        heatmap = self.grad_cam.compute_heatmap(self.test_image)
        
        # Check heatmap shape (should be a 2D array)
        self.assertEqual(len(heatmap.shape), 2)
        
        # Check values are normalized (between 0 and 1)
        self.assertTrue(np.all(heatmap >= 0) and np.all(heatmap <= 1))

    def test_overlay_heatmap(self):
        """Test heatmap overlay"""
        # Create a simple heatmap
        heatmap = np.zeros((10, 10))
        heatmap[3:7, 3:7] = 1.0  # Create a hot spot in the center
        
        # Create a sample image
        image = np.ones((10, 10)) * 128  # Gray image
        image = image.astype(np.uint8)
        
        # Generate overlay
        overlaid = self.grad_cam.overlay_heatmap(image, heatmap)
        
        # Check overlay shape and channels (should be RGB)
        self.assertEqual(len(overlaid.shape), 3)
        self.assertEqual(overlaid.shape[2], 3)
        
        # Check values are in valid range for uint8
        self.assertTrue(np.all(overlaid >= 0) and np.all(overlaid <= 255))

    def test_generate_visualization(self):
        """Test full visualization generation"""
        # Generate visualization
        orig, heatmap, overlaid = self.grad_cam.generate_visualization(self.test_image)
        
        # Check outputs
        self.assertIsNotNone(orig)
        self.assertIsNotNone(heatmap)
        self.assertIsNotNone(overlaid)
        
        # Check types and shapes
        self.assertEqual(orig.dtype, np.uint8)
        self.assertTrue(np.all(heatmap >= 0) and np.all(heatmap <= 1))
        self.assertEqual(overlaid.shape[2], 3)  # RGB image

if __name__ == '__main__':
    unittest.main() 