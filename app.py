import streamlit as st
import cv2
import numpy as np
from src.model import XRayAnalyzer
import matplotlib.pyplot as plt
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Medical X-ray Analyzer",
    page_icon="🏥",
    layout="wide"
)

# Initialize the model
@st.cache_resource
def load_model():
    return XRayAnalyzer()

# Title and description
st.title("Medical X-ray Analysis System")
st.markdown("""
This application uses advanced computer vision and machine learning to analyze medical X-ray images.
Upload an X-ray image to get a detailed analysis including:
- Normal/Abnormal classification
- Confidence scores
- Region of interest highlighting
- Detailed probability distribution
""")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)
    
    # Convert PIL Image to numpy array for processing
    image_np = np.array(image)
    
    # Get model predictions
    model = load_model()
    results = model.analyze(image_np)
    
    # Create two columns for results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Analysis Results")
        st.write(f"Prediction: **{results['prediction']}**")
        st.write(f"Confidence: **{results['confidence']*100:.2f}%**")
        
        # Display probability distribution
        probs = results['probabilities']
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(['Normal', 'Abnormal'], 
               [probs['normal'], probs['abnormal']],
               color=['green', 'red'])
        ax.set_ylim(0, 1)
        ax.set_title('Probability Distribution')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Region of Interest")
        # Get attention map
        attention_map = results['attention_map']
        
        # Create heatmap
        heatmap = cv2.applyColorMap(
            (attention_map * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        # Resize heatmap to match original image
        heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        overlay = cv2.addWeighted(image_np, 0.7, heatmap, 0.3, 0)
        
        st.image(overlay, caption="Regions of Interest Highlighted", use_column_width=True)
    
    # Additional information
    st.markdown("---")
    st.markdown("""
    ### Interpretation Guide
    - **Normal**: No significant abnormalities detected
    - **Abnormal**: Potential medical conditions detected
    - The highlighted regions indicate areas of interest that the model focused on
    - Confidence scores indicate the model's certainty in its prediction
    """)
    
    # Disclaimer
    st.markdown("""
    ---
    ⚠️ **Disclaimer**: This tool is for educational and research purposes only. 
    Always consult with qualified medical professionals for proper diagnosis and treatment.
    """) 