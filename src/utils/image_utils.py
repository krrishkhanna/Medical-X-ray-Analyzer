"""
Utility functions for image processing
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from typing import Tuple, List, Optional

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from path
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded image as numpy array
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    return image

def save_image(image: np.ndarray, save_path: str) -> None:
    """
    Save an image to disk
    
    Args:
        image: Image to save
        save_path: Path to save the image
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    # Save the image
    cv2.imwrite(save_path, image)

def visualize_results(original_image: np.ndarray, processed_image: np.ndarray, 
                     heatmap: np.ndarray, prediction: float,
                     save_path: Optional[str] = None) -> None:
    """
    Visualize original image, processed image, and heatmap
    
    Args:
        original_image: Original input image
        processed_image: Processed image
        heatmap: GradCAM heatmap
        prediction: Model prediction
        save_path: Path to save the visualization
    """
    # Create a figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    if len(original_image.shape) == 2:
        axs[0].imshow(original_image, cmap='gray')
    else:
        axs[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    # Plot processed image
    axs[1].imshow(processed_image, cmap='gray')
    axs[1].set_title('Processed Image')
    axs[1].axis('off')
    
    # Plot heatmap
    axs[2].imshow(heatmap)
    axs[2].set_title(f'GradCAM Heatmap (Pred: {prediction:.2f})')
    axs[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def batch_process_images(input_dir: str, output_dir: str, 
                         processor, extension: str = '*.jpg') -> None:
    """
    Batch process all images in a directory
    
    Args:
        input_dir: Directory containing images
        output_dir: Directory to save processed images
        processor: Function to process each image
        extension: File extension to filter by
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    input_path = Path(input_dir)
    image_files = list(input_path.glob(extension))
    
    print(f"Processing {len(image_files)} images...")
    
    # Process each image
    for img_path in image_files:
        # Load image
        image = load_image(str(img_path))
        
        # Process image
        processed = processor(image)
        
        # Save processed image
        output_path = os.path.join(output_dir, img_path.name)
        save_image(processed, output_path)
        
    print(f"Processing complete. Results saved to {output_dir}")

def create_comparison_grid(images: List[np.ndarray], titles: List[str], 
                          rows: int, cols: int, figsize: Tuple[int, int] = (15, 10),
                          save_path: Optional[str] = None) -> None:
    """
    Create a grid of images for comparison
    
    Args:
        images: List of images to display
        titles: List of titles for each image
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        figsize: Figure size
        save_path: Path to save the grid
    """
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.flatten()
    
    for i, (img, title) in enumerate(zip(images, titles)):
        if i < len(axs):
            if len(img.shape) == 2:
                axs[i].imshow(img, cmap='gray')
            else:
                axs[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[i].set_title(title)
            axs[i].axis('off')
    
    # Hide any unused subplots
    for i in range(len(images), len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close() 