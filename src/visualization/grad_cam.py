"""
Grad-CAM visualization module for model interpretability
"""

import tensorflow as tf
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM
        
        Args:
            model: Trained model
            layer_name: Name of the layer to compute Grad-CAM on (defaults to last conv layer)
        """
        self.model = model
        
        # If layer name not provided, get the last conv layer
        if layer_name is None:
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    layer_name = layer.name
                    break
        
        self.layer_name = layer_name
        self.grad_model = self._create_grad_model()

    def _create_grad_model(self):
        """
        Create a model that outputs both the predictions and the activations of the target layer
        
        Returns:
            Keras model
        """
        grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )
        return grad_model

    def compute_heatmap(self, image, class_idx=None, eps=1e-8):
        """
        Compute Grad-CAM heatmap
        
        Args:
            image: Input image (should be preprocessed)
            class_idx: Index of the class to generate heatmap for (defaults to predicted class)
            eps: Small constant to avoid division by zero
            
        Returns:
            Normalized heatmap
        """
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = tf.expand_dims(image, 0)
            
        # Get gradients and activations
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            class_channel = predictions[:, class_idx]
            
        # Calculate gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Calculate guided gradients
        guided_grads = tf.cast(conv_outputs > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
        
        # Calculate weights
        weights = tf.reduce_mean(guided_grads, axis=(1, 2))
        
        # Generate heatmap
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
        
        # Normalize heatmap
        heatmap = tf.maximum(cam, 0) / (tf.reduce_max(cam) + eps)
        
        return heatmap.numpy()[0]

    def overlay_heatmap(self, image, heatmap, alpha=0.4):
        """
        Overlay heatmap on original image
        
        Args:
            image: Original image
            heatmap: Computed heatmap
            alpha: Transparency factor
            
        Returns:
            Overlaid image
        """
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert image to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        # Ensure image is in 0-255 range
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        # Overlay heatmap on image
        overlaid = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
        
        return overlaid

    def generate_visualization(self, image, class_idx=None):
        """
        Generate complete Grad-CAM visualization
        
        Args:
            image: Input image
            class_idx: Index of the class to generate visualization for
            
        Returns:
            Original image, heatmap, and overlaid image
        """
        # Compute heatmap
        heatmap = self.compute_heatmap(image, class_idx)
        
        # Get original image (before preprocessing)
        orig_image = (image.numpy()[0] * 255).astype(np.uint8)
        
        # Create overlay
        overlaid = self.overlay_heatmap(orig_image, heatmap)
        
        return orig_image, heatmap, overlaid 