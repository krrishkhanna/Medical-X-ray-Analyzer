"""
Script for preparing and processing the raw dataset
"""

import os
import shutil
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from ..preprocessing.image_processor import XRayPreprocessor
from ..config import DATA_CONFIG

def prepare_mura_dataset():
    """
    Prepare MURA dataset for training
    - Organize files into train/val/test splits
    - Preprocess images
    """
    print("Preparing MURA dataset...")
    
    # Load CSV files
    train_csv = os.path.join(DATA_CONFIG['MURA_DATASET_PATH'], 'MURA-v1.1/train_image_paths.csv')
    valid_csv = os.path.join(DATA_CONFIG['MURA_DATASET_PATH'], 'MURA-v1.1/valid_image_paths.csv')
    
    if not os.path.exists(train_csv) or not os.path.exists(valid_csv):
        print(f"Error: CSV files not found at {train_csv} or {valid_csv}")
        return
    
    train_df = pd.read_csv(train_csv, names=['path'])
    valid_df = pd.read_csv(valid_csv, names=['path'])
    
    # Create output directories
    output_dir = os.path.join(DATA_CONFIG['PROCESSED_DATA_PATH'], 'mura')
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Create positive and negative subdirectories
    os.makedirs(os.path.join(train_dir, 'positive'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'negative'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'positive'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'negative'), exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = XRayPreprocessor()
    
    # Process training images
    print("Processing training images...")
    _process_images(train_df, train_dir, preprocessor)
    
    # Process validation images
    print("Processing validation images...")
    _process_images(valid_df, val_dir, preprocessor)
    
    print(f"MURA dataset preparation complete. Processed data saved to {output_dir}")

def prepare_chestxray_dataset():
    """
    Prepare ChestX-ray14 dataset for training
    - Filter for pneumonia cases
    - Split into train/val/test
    - Preprocess images
    """
    print("Preparing ChestX-ray14 dataset...")
    
    # Load data entry file
    data_entry_path = os.path.join(DATA_CONFIG['CHESTXRAY_DATASET_PATH'], 'Data_Entry_2017.csv')
    if not os.path.exists(data_entry_path):
        print(f"Error: Data entry file not found at {data_entry_path}")
        return
    
    df = pd.read_csv(data_entry_path)
    
    # Filter for pneumonia and non-pneumonia cases
    pneumonia_df = df[df['Finding Labels'].str.contains('Pneumonia') | 
                     df['Finding Labels'].str.contains('Tuberculosis')]
    
    # Sample an equal number of normal cases
    normal_df = df[df['Finding Labels'] == 'No Finding'].sample(n=len(pneumonia_df), random_state=42)
    
    # Combine and shuffle
    combined_df = pd.concat([pneumonia_df, normal_df]).sample(frac=1, random_state=42)
    
    # Create labels
    combined_df['label'] = combined_df['Finding Labels'].apply(
        lambda x: 'positive' if 'Pneumonia' in x or 'Tuberculosis' in x else 'negative'
    )
    
    # Create train/val/test split
    train_df = combined_df.sample(frac=DATA_CONFIG['TRAIN_SPLIT'], random_state=42)
    remaining_df = combined_df.drop(train_df.index)
    
    val_test_split = DATA_CONFIG['VAL_SPLIT'] / (DATA_CONFIG['VAL_SPLIT'] + DATA_CONFIG['TEST_SPLIT'])
    val_df = remaining_df.sample(frac=val_test_split, random_state=42)
    test_df = remaining_df.drop(val_df.index)
    
    # Create output directories
    output_dir = os.path.join(DATA_CONFIG['PROCESSED_DATA_PATH'], 'chestxray')
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Create positive and negative subdirectories
    os.makedirs(os.path.join(train_dir, 'positive'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'negative'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'positive'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'negative'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'positive'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'negative'), exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = XRayPreprocessor()
    
    # Process training images
    print("Processing training images...")
    _process_chestxray_images(train_df, train_dir, preprocessor)
    
    # Process validation images
    print("Processing validation images...")
    _process_chestxray_images(val_df, val_dir, preprocessor)
    
    # Process test images
    print("Processing test images...")
    _process_chestxray_images(test_df, test_dir, preprocessor)
    
    print(f"ChestX-ray14 dataset preparation complete. Processed data saved to {output_dir}")

def _process_images(df, output_dir, preprocessor):
    """Process MURA images and save to output directory"""
    for i, row in tqdm(df.iterrows(), total=len(df)):
        # Get image path
        img_path = os.path.join(DATA_CONFIG['MURA_DATASET_PATH'], row['path'])
        
        # Determine label
        label = 'positive' if 'positive' in img_path else 'negative'
        
        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        
        # Preprocess image
        processed_img = preprocessor.preprocess_image(img)
        
        # Generate filename
        filename = f"{Path(img_path).stem}.png"
        output_path = os.path.join(output_dir, label, filename)
        
        # Save processed image
        cv2.imwrite(output_path, (processed_img * 255).astype(np.uint8))

def _process_chestxray_images(df, output_dir, preprocessor):
    """Process ChestX-ray14 images and save to output directory"""
    for i, row in tqdm(df.iterrows(), total=len(df)):
        # Get image path
        img_path = os.path.join(DATA_CONFIG['CHESTXRAY_DATASET_PATH'], 'images', row['Image Index'])
        
        # Get label
        label = row['label']
        
        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        
        # Preprocess image
        processed_img = preprocessor.preprocess_image(img)
        
        # Generate filename
        filename = f"{Path(img_path).stem}.png"
        output_path = os.path.join(output_dir, label, filename)
        
        # Save processed image
        cv2.imwrite(output_path, (processed_img * 255).astype(np.uint8))

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Prepare dataset for training')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['mura', 'chestxray', 'all'],
        default='all',
        help='Dataset to prepare'
    )
    args = parser.parse_args()
    
    # Create processed data directory
    os.makedirs(DATA_CONFIG['PROCESSED_DATA_PATH'], exist_ok=True)
    
    if args.dataset == 'mura' or args.dataset == 'all':
        prepare_mura_dataset()
    
    if args.dataset == 'chestxray' or args.dataset == 'all':
        prepare_chestxray_dataset()

if __name__ == '__main__':
    main() 