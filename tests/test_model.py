"""
Unit tests for XRay model
"""

import unittest
import numpy as np
import tensorflow as tf
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.xray_model import XRayModel

class TestModel(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.model = XRayModel()
        # Create a sample test image batch (with proper dimensions)
        self.test_image = np.random.rand(1, 224, 224, 1).astype(np.float32)

    def test_model_initialization(self):
        """Test model initialization"""
        # Check that the model was created successfully
        self.assertIsNotNone(self.model.model)
        
        # Check model input shape
        self.assertEqual(self.model.model.input_shape[1:3], (224, 224))
        
        # Check model output shape matches num_classes
        from src.config import MODEL_CONFIG
        self.assertEqual(self.model.model.output_shape[1], MODEL_CONFIG['NUM_CLASSES'])

    def test_prediction(self):
        """Test model prediction"""
        # Get prediction
        prediction = self.model.predict(self.test_image)
        
        # Check prediction shape and type
        self.assertEqual(prediction.shape[0], 1)  # Batch size of 1
        from src.config import MODEL_CONFIG
        self.assertEqual(prediction.shape[1], MODEL_CONFIG['NUM_CLASSES'])
        
        # Check prediction values are between 0 and 1 (sigmoid output)
        self.assertTrue(np.all(prediction >= 0) and np.all(prediction <= 1))

    def test_callbacks(self):
        """Test that callbacks are created properly"""
        callbacks = self.model._create_callbacks()
        
        # Check that we have the expected number of callbacks
        self.assertEqual(len(callbacks), 3)
        
        # Check types of callbacks
        callback_types = [type(callback).__name__ for callback in callbacks]
        self.assertIn('EarlyStopping', callback_types)
        self.assertIn('ModelCheckpoint', callback_types)
        self.assertIn('ReduceLROnPlateau', callback_types)

    def test_save_load(self):
        """Test model save and load functionality"""
        # Create a temporary directory for saving the model
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, 'test_model.h5')
            
            # Save the model
            self.model.save(model_path)
            
            # Check that the file exists
            self.assertTrue(os.path.exists(model_path))
            
            # Try to load the model
            try:
                loaded_model = XRayModel.load(model_path)
                self.assertIsNotNone(loaded_model.model)
            except Exception as e:
                self.fail(f"Model loading failed with error: {str(e)}")

if __name__ == '__main__':
    unittest.main() 