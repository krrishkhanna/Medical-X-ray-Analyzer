"""
X-ray classification model implementation
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from ..config import MODEL_CONFIG

class XRayModel:
    def __init__(self):
        """Initialize the X-ray classification model"""
        self.model = self._build_model()
        self.callbacks = self._create_callbacks()

    def _build_model(self):
        """
        Build and compile the model architecture
        
        Returns:
            Compiled Keras model
        """
        # Choose base model
        if MODEL_CONFIG['MODEL_TYPE'] == 'mobilenetv2':
            base_model = applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            )
        else:  # ResNet18
            base_model = applications.ResNet50(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Create model
        model = models.Sequential([
            # Convert single channel to 3 channels for pretrained model
            layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x)),
            
            # Base model
            base_model,
            
            # Additional layers
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(MODEL_CONFIG['NUM_CLASSES'], activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=MODEL_CONFIG['LEARNING_RATE']),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model

    def _create_callbacks(self):
        """
        Create training callbacks
        
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=MODEL_CONFIG['EARLY_STOPPING_PATIENCE'],
                restore_best_weights=True
            ),
            
            # Model checkpoint
            ModelCheckpoint(
                filepath=f"{MODEL_CONFIG['CHECKPOINT_PATH']}/model_best.h5",
                monitor='val_loss',
                save_best_only=True
            ),
            
            # Learning rate scheduler
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6
            )
        ]
        return callbacks

    def train(self, train_ds, val_ds, class_weights=None):
        """
        Train the model
        
        Args:
            train_ds: Training dataset
            val_ds: Validation dataset
            class_weights: Optional class weights for imbalanced datasets
            
        Returns:
            Training history
        """
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=MODEL_CONFIG['EPOCHS'],
            callbacks=self.callbacks,
            class_weight=class_weights
        )
        return history

    def evaluate(self, test_ds):
        """
        Evaluate the model
        
        Args:
            test_ds: Test dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = self.model.evaluate(test_ds, return_dict=True)
        return results

    def predict(self, image):
        """
        Make prediction for a single image
        
        Args:
            image: Preprocessed image tensor
            
        Returns:
            Prediction probabilities
        """
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = tf.expand_dims(image, 0)
        
        predictions = self.model.predict(image)
        return predictions

    def save(self, filepath):
        """Save the model to disk"""
        self.model.save(filepath)

    @classmethod
    def load(cls, filepath):
        """Load a saved model from disk"""
        model = cls()
        model.model = tf.keras.models.load_model(filepath)
        return model 