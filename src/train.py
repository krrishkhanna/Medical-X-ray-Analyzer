"""
Main training script for X-ray analysis model
"""

import argparse
import logging
from pathlib import Path

from data.data_loader import XRayDataLoader
from models.xray_model import XRayModel
from config import MODEL_CONFIG

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train X-ray analysis model')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['mura', 'chestxray'],
        default='mura',
        help='Dataset to use for training'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['mobilenetv2', 'resnet18'],
        default='mobilenetv2',
        help='Type of model architecture to use'
    )
    return parser.parse_args()

def main():
    """Main training function"""
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()
    
    logger.info(f"Starting training with {args.dataset} dataset and {args.model_type} architecture")
    
    # Create checkpoint directory
    Path(MODEL_CONFIG['CHECKPOINT_PATH']).mkdir(parents=True, exist_ok=True)
    
    try:
        # Load dataset
        logger.info("Loading dataset...")
        data_loader = XRayDataLoader(dataset_type=args.dataset)
        train_ds, val_ds, test_ds = data_loader.load_dataset()
        
        # Calculate class weights
        class_weights = None
        if hasattr(train_ds, 'labels'):
            class_weights = data_loader.get_class_weights(train_ds.labels)
        
        # Create and train model
        logger.info("Creating model...")
        model = XRayModel()
        
        logger.info("Starting training...")
        history = model.train(train_ds, val_ds, class_weights=class_weights)
        
        # Evaluate model
        logger.info("Evaluating model...")
        results = model.evaluate(test_ds)
        
        # Log results
        logger.info("Training completed. Test set metrics:")
        for metric, value in results.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Save model
        save_path = f"{MODEL_CONFIG['CHECKPOINT_PATH']}/model_final.h5"
        model.save(save_path)
        logger.info(f"Model saved to {save_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 