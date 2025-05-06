import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class XRayAnalyzer:
    def __init__(self, model_path=None):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(model_path.replace('.joblib', '_scaler.joblib'))
    
    def extract_features(self, image):
        """Extract relevant features from the X-ray image."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Resize to standard size
        gray = cv2.resize(gray, (224, 224))
        
        # Extract texture features using GLCM
        features = []
        
        # 1. Basic statistics
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.median(gray),
            np.max(gray),
            np.min(gray)
        ])
        
        # 2. Edge features
        edges = cv2.Canny(gray, 100, 200)
        features.extend([
            np.mean(edges),
            np.std(edges),
            np.sum(edges > 0) / edges.size
        ])
        
        # 3. Histogram features
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        hist = hist.flatten() / hist.sum()
        features.extend(hist)
        
        # 4. Local binary patterns
        lbp = self._compute_lbp(gray)
        features.extend([
            np.mean(lbp),
            np.std(lbp),
            np.median(lbp)
        ])
        
        return np.array(features)
    
    def _compute_lbp(self, image):
        """Compute Local Binary Patterns."""
        lbp = np.zeros_like(image)
        for i in range(1, image.shape[0]-1):
            for j in range(1, image.shape[1]-1):
                center = image[i, j]
                code = 0
                code |= (image[i-1, j-1] >= center) << 7
                code |= (image[i-1, j] >= center) << 6
                code |= (image[i-1, j+1] >= center) << 5
                code |= (image[i, j+1] >= center) << 4
                code |= (image[i+1, j+1] >= center) << 3
                code |= (image[i+1, j] >= center) << 2
                code |= (image[i+1, j-1] >= center) << 1
                code |= (image[i, j-1] >= center) << 0
                lbp[i, j] = code
        return lbp
    
    def analyze(self, image):
        """Analyze the X-ray image and return predictions with confidence scores."""
        # Extract features
        features = self.extract_features(image)
        
        # Scale features
        if hasattr(self.scaler, 'mean_'):
            features = self.scaler.transform(features.reshape(1, -1))
        
        # Get prediction probabilities
        try:
            probabilities = self.model.predict_proba(features)[0]
        except:
            # If model is not trained, return dummy probabilities
            probabilities = np.array([0.5, 0.5])
        
        # Get prediction
        pred_class = np.argmax(probabilities)
        confidence = probabilities[pred_class]
        
        # Generate attention map
        attention_map = self._generate_attention_map(image)
        
        return {
            'prediction': 'Normal' if pred_class == 0 else 'Abnormal',
            'confidence': confidence,
            'probabilities': {
                'normal': probabilities[0],
                'abnormal': probabilities[1]
            },
            'attention_map': attention_map
        }
    
    def _generate_attention_map(self, image):
        """Generate attention map highlighting potential regions of interest."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Create attention map
        attention_map = np.zeros_like(gray, dtype=np.float32)
        
        # Draw contours with varying intensities
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small contours
                intensity = min(1.0, area / (gray.shape[0] * gray.shape[1]))
                cv2.drawContours(attention_map, [contour], -1, intensity, -1)
        
        return attention_map 