import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import cv2
from PIL import Image
import os
from pathlib import Path

class EnhancedXRayDemo:
    def __init__(self):
        """Initialize the enhanced demo app"""
        # Create upload directory if it doesn't exist
        os.makedirs('uploads', exist_ok=True)
        
        # Configure page
        st.set_page_config(
            page_title="Medical X-ray Analyzer",
            page_icon="🩻",
            layout="wide"
        )
        
    def preprocess_image(self, image):
        """Simulate image preprocessing pipeline"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Resize image
        resized = cv2.resize(gray, (224, 224))
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)
        
        # Normalize pixel values
        normalized = enhanced.astype(np.float32) / 255.0
        
        return normalized, enhanced
    
    def detect_edges(self, image):
        """Detect edges in the image using Canny edge detector"""
        # Auto-calculate thresholds
        sigma = 0.33
        median = np.median(image)
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        
        edges = cv2.Canny(image, lower, upper)
        return edges
    
    def create_simulated_heatmap(self, image):
        """Create a simulated heatmap based on image features"""
        # Use Gaussian blur to create a focus area
        blurred = cv2.GaussianBlur(image, (51, 51), 0)
        
        # Find the brightest region (simplified approach)
        _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
        
        # Create a heatmap using distance from the brightest area
        heatmap = np.zeros_like(image)
        
        # Simulating a region of interest in the center
        center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                # Distance from center
                dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                # Inverse of distance (closer = hotter)
                val = max(0, 255 - dist * 1.5)
                heatmap[y, x] = val
                
        # Normalize and apply colormap
        heatmap = (heatmap / 255.0)
        # Add some randomness to make it look more realistic
        heatmap = heatmap * (0.8 + 0.4 * np.random.rand(heatmap.shape[0], heatmap.shape[1]))
        
        return heatmap
    
    def simulate_prediction(self, image):
        """Simulate model prediction"""
        # Extract some features to simulate prediction
        edges = self.detect_edges(image)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Simulate probability based on image features
        base_probability = 0.2 + 0.6 * np.random.rand()  # Random baseline
        feature_contribution = edge_density * 0.5  # Edge density contribution
        
        # Combine for final probability
        probability = min(0.95, max(0.05, base_probability + feature_contribution))
        
        return probability
    
    def run(self):
        """Run the enhanced demo app"""
        st.title("🩻 Medical X-ray Analyzer")
        st.write("""
        Upload an X-ray image to analyze it for potential medical conditions.
        
        This enhanced demo simulates the full X-ray analysis pipeline, including:
        - Image preprocessing (CLAHE, edge detection)
        - Feature extraction
        - Abnormality detection
        - Region-of-interest visualization
        """)
        
        # Add sidebar
        with st.sidebar:
            st.header("About")
            st.info("""
            This is a demonstration of medical X-ray analysis using computer vision.
            
            In a full implementation, this would use deep learning models like MobileNetV2
            or ResNet18 trained on MURA or ChestX-ray14 datasets.
            """)
            
            st.header("Settings")
            model_type = st.selectbox(
                "Model Architecture (simulated)",
                ["MobileNetV2", "ResNet18"]
            )
            
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Bone Fracture Detection", "Pneumonia Detection"]
            )
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an X-ray image",
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            save_path = Path('uploads') / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Read image
            image = cv2.imread(str(save_path))
            
            if image is None:
                st.error("Error reading the image. Please try another file.")
                return
            
            # Create two columns
            col1, col2 = st.columns(2)
            
            # Display original image
            with col1:
                st.subheader("Original X-ray")
                st.image(uploaded_file, use_column_width=True)
            
            # Process button
            if st.button("Analyze X-ray"):
                # Simulate processing
                with st.spinner("Analyzing X-ray image..."):
                    import time
                    time.sleep(2)  # Simulate processing time
                    
                    # Preprocess image
                    normalized, enhanced = self.preprocess_image(image)
                    
                    # Detect edges
                    edges = self.detect_edges(enhanced)
                    
                    # Create heatmap
                    heatmap = self.create_simulated_heatmap(enhanced)
                    
                    # Simulate prediction
                    probability = self.simulate_prediction(enhanced)
                    condition_detected = probability > 0.5
                    
                    # Create overlay
                    color_heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(
                        cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), 0.7,
                        color_heatmap, 0.3, 0
                    )
                    
                    # Display results in second column
                    with col2:
                        st.subheader("Analysis Results")
                        
                        # Display prediction
                        st.write("### Diagnosis")
                        if condition_detected:
                            st.error(f"⚠️ Potential abnormality detected")
                            st.write(f"Confidence: {probability:.2%}")
                        else:
                            st.success("✅ No significant abnormalities detected")
                            st.write(f"Confidence: {1-probability:.2%}")
                        
                        # Display processing stages
                        st.write("### Processing pipeline")
                        
                        # Create tabs for different views
                        tab1, tab2, tab3 = st.tabs(["Region Analysis", "Preprocessed Image", "Edge Detection"])
                        
                        with tab1:
                            st.image(
                                cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), 
                                caption="Region of Interest", 
                                use_column_width=True
                            )
                            st.info("The heatmap highlights regions that influenced the analysis.")
                            
                        with tab2:
                            st.image(
                                enhanced, 
                                caption="Preprocessed with CLAHE", 
                                use_column_width=True
                            )
                            
                        with tab3:
                            st.image(
                                edges, 
                                caption="Edge Detection", 
                                use_column_width=True
                            )
                        
                        # Add explanation
                        st.write("### Explanation")
                        st.write(f"""
                        Using simulated {model_type} analysis for {analysis_type.lower()}.
                        
                        This demo detected {'abnormal patterns' if condition_detected else 'normal patterns'} 
                        in the image. In a full implementation, this would use a trained deep learning model 
                        to provide accurate medical analysis.
                        """)
                
                # Show feature analysis section
                st.subheader("Feature Analysis")
                
                # Create a collapsible section with technical details
                with st.expander("See technical details"):
                    # Display histogram
                    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                    
                    # Original image histogram
                    ax[0].hist(image.ravel(), 256, [0, 256])
                    ax[0].set_title('Original Histogram')
                    ax[0].set_xlabel('Pixel Value')
                    ax[0].set_ylabel('Frequency')
                    
                    # Enhanced image histogram
                    ax[1].hist(enhanced.ravel(), 256, [0, 256])
                    ax[1].set_title('Enhanced Histogram')
                    ax[1].set_xlabel('Pixel Value')
                    ax[1].set_ylabel('Frequency')
                    
                    # Display figure
                    st.pyplot(fig)
                    
                    # Display feature info
                    st.write(f"""
                    - Edge density: {np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]):.2%}
                    - Mean pixel value: {np.mean(enhanced):.2f}
                    - Standard deviation: {np.std(enhanced):.2f}
                    """)
            
            # Clean up
            os.remove(save_path)

def main():
    """Main function"""
    app = EnhancedXRayDemo()
    app.run()

if __name__ == "__main__":
    main() 