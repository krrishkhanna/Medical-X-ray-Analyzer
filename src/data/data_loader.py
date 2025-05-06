"""
Data loader module for X-ray datasets
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from ..config import DATA_CONFIG
from ..preprocessing.image_processor import XRayPreprocessor

class XRayDataLoader:
    def __init__(self, dataset_type='mura'):
        """
        Initialize data loader
        
        Args:
            dataset_type: Type of dataset to load ('mura' or 'chestxray')
        """
        self.dataset_type = dataset_type
        self.preprocessor = XRayPreprocessor()
        self.data_path = (DATA_CONFIG['MURA_DATASET_PATH'] 
                         if dataset_type == 'mura' 
                         else DATA_CONFIG['CHESTXRAY_DATASET_PATH'])

    def load_dataset(self):
        """
        Load and prepare the dataset
        
        Returns:
            train_ds, val_ds, test_ds: TensorFlow dataset objects
        """
        # Load image paths and labels
        image_paths, labels = self._get_dataset_info()
        
        # Split dataset
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            image_paths, labels,
            test_size=1-DATA_CONFIG['TRAIN_SPLIT'],
            stratify=labels,
            random_state=42
        )
        
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            test_paths, test_labels,
            test_size=DATA_CONFIG['TEST_SPLIT']/(DATA_CONFIG['TEST_SPLIT'] + DATA_CONFIG['VAL_SPLIT']),
            stratify=test_labels,
            random_state=42
        )
        
        # Create TensorFlow datasets
        train_ds = self._create_tf_dataset(train_paths, train_labels, augment=True)
        val_ds = self._create_tf_dataset(val_paths, val_labels, augment=False)
        test_ds = self._create_tf_dataset(test_paths, test_labels, augment=False)
        
        return train_ds, val_ds, test_ds

    def _get_dataset_info(self):
        """
        Get image paths and labels from dataset
        
        Returns:
            image_paths: List of image file paths
            labels: List of corresponding labels
        """
        image_paths = []
        labels = []
        
        if self.dataset_type == 'mura':
            # Load MURA dataset
            csv_path = os.path.join(self.data_path, 'MURA-v1.1/train_image_paths.csv')
            df = pd.read_csv(csv_path)
            image_paths = df['Path'].tolist()
            labels = [1 if 'positive' in path else 0 for path in image_paths]
        else:
            # Load ChestX-ray14 dataset
            data_entry_path = os.path.join(self.data_path, 'Data_Entry_2017.csv')
            df = pd.read_csv(data_entry_path)
            image_paths = df['Image Index'].apply(lambda x: os.path.join(self.data_path, 'images', x)).tolist()
            labels = (df['Finding Labels'].str.contains('Pneumonia') | 
                     df['Finding Labels'].str.contains('Tuberculosis')).astype(int).tolist()
        
        return image_paths, labels

    def _create_tf_dataset(self, image_paths, labels, augment=False):
        """
        Create a TensorFlow dataset from image paths and labels
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            augment: Whether to apply data augmentation
            
        Returns:
            TensorFlow dataset object
        """
        def load_and_preprocess(path, label):
            # Load image
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=1)
            
            # Convert to numpy for preprocessing
            img_np = img.numpy()
            
            # Apply preprocessing
            img_processed = self.preprocessor.preprocess_image(img_np, augment=augment)
            
            return tf.convert_to_tensor(img_processed, dtype=tf.float32), label

        # Create dataset
        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(DATA_CONFIG['BATCH_SIZE'])
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds

    def get_class_weights(self, labels):
        """
        Calculate class weights for imbalanced datasets
        
        Args:
            labels: List of labels
            
        Returns:
            Dictionary of class weights
        """
        total = len(labels)
        class_counts = np.bincount(labels)
        class_weights = {i: total / (len(class_counts) * count) 
                        for i, count in enumerate(class_counts)}
        return class_weights 