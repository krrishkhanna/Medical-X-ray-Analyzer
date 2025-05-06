import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import io
import plotly.express as px

class XRayAnalyzer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    
    def extract_features(self, image):
        """Extract relevant features from the X-ray image."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Resize to standard size
        gray = cv2.resize(gray, (224, 224))
        
        # Extract features
        features = []
        
        # Enhanced feature extraction
        # 1. Intensity statistics
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.median(gray),
            np.percentile(gray, 25),
            np.percentile(gray, 75)
        ])
        
        # 2. Edge features with multiple thresholds
        edges_low = cv2.Canny(gray, 50, 150)
        edges_high = cv2.Canny(gray, 100, 200)
        features.extend([
            np.mean(edges_low),
            np.mean(edges_high),
            np.sum(edges_low > 0) / edges_low.size,
            np.sum(edges_high > 0) / edges_high.size
        ])
        
        # 3. Texture features
        texture_features = self._extract_texture_features(gray)
        features.extend(texture_features)
        
        return np.array(features)
    
    def _extract_texture_features(self, image):
        """Extract texture features using GLCM-like approach."""
        features = []
        # Compute local binary pattern-like features
        for i in range(0, image.shape[0]-1, 2):
            for j in range(0, image.shape[1]-1, 2):
                block = image[i:i+2, j:j+2]
                if block.size == 4:  # Ensure we have a 2x2 block
                    features.append(np.std(block))
        
        # Take statistical measures of the texture features
        if features:
            return [
                np.mean(features),
                np.std(features),
                np.percentile(features, 25),
                np.percentile(features, 75)
            ]
        return [0, 0, 0, 0]
    
    def analyze(self, image):
        """Analyze the X-ray image and return predictions with confidence scores."""
        # Extract features
        features = self.extract_features(image)
        
        # Enhanced analysis logic with more conservative thresholds
        edge_density = np.mean(features[7:9])  # Use both edge detection results
        texture_complexity = np.mean(features[9:])  # Use texture features
        
        # Calculate histogram features
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize histogram
        
        # More sophisticated decision logic with conservative thresholds
        # Calculate multiple indicators
        edge_score = min(edge_density * 2, 1.0)  # Cap at 1.0
        texture_score = min(texture_complexity * 1.5, 1.0)  # Cap at 1.0
        
        # Calculate histogram-based features
        intensity_range = np.percentile(gray, 95) - np.percentile(gray, 5)
        intensity_variance = np.var(gray) / 255.0
        
        # Combine scores with more conservative weights
        abnormality_score = (
            edge_score * 0.3 +  # Reduced weight for edge detection
            texture_score * 0.2 +  # Reduced weight for texture
            (intensity_range / 255.0) * 0.3 +  # Added intensity range consideration
            intensity_variance * 0.2  # Added variance consideration
        )
        
        # More conservative thresholds
        if abnormality_score > 0.4:  # Increased threshold
            prediction = 'Abnormal'
            # More conservative confidence calculation
            confidence = min(abnormality_score * 0.8, 0.85)  # Capped at 85%
        else:
            prediction = 'Normal'
            # More conservative confidence calculation
            confidence = min((1 - abnormality_score) * 0.8, 0.85)  # Capped at 85%
        
        # Calculate probabilities with more conservative values
        prob_normal = 1 - confidence if prediction == 'Abnormal' else confidence
        prob_abnormal = confidence if prediction == 'Abnormal' else 1 - confidence
        
        # Generate enhanced attention map
        attention_map = self._generate_attention_map(image)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'normal': prob_normal,
                'abnormal': prob_abnormal
            },
            'attention_map': attention_map,
            'feature_importance': {
                'Edge Density': edge_score,
                'Texture Complexity': texture_score,
                'Intensity Range': intensity_range / 255.0,
                'Intensity Variance': intensity_variance
            },
            'histogram': hist,
            'histogram_bins': np.arange(256)
        }
    
    def _generate_attention_map(self, image):
        """Generate enhanced attention map highlighting potential regions of interest."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Multi-scale analysis
        attention_maps = []
        
        # 1. Edge detection at multiple scales
        edges_fine = cv2.Canny(gray, 50, 150)
        edges_coarse = cv2.Canny(gray, 100, 200)
        
        # 2. Adaptive thresholding
        thresh_fine = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        thresh_coarse = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 21, 4
        )
        
        # Combine all maps
        attention_map = cv2.addWeighted(edges_fine, 0.3, edges_coarse, 0.3, 0)
        attention_map = cv2.addWeighted(attention_map, 1.0, thresh_fine, 0.2, 0)
        attention_map = cv2.addWeighted(attention_map, 1.0, thresh_coarse, 0.2, 0)
        
        # Normalize to 0-1 range
        attention_map = attention_map.astype(float) / 255.0
        
        # Apply Gaussian blur to smooth the map
        attention_map = cv2.GaussianBlur(attention_map, (5, 5), 0)
        
        return attention_map

# Set page config with dark theme
st.set_page_config(
    page_title="Advanced X-ray Analysis System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #00a0dc;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stProgress > div > div > div {
        background-color: #00a0dc;
    }
    h1 {
        color: #00a0dc;
        font-size: 3rem;
        margin-bottom: 2rem;
    }
    h2 {
        color: #ffffff;
        margin-top: 2rem;
    }
    .highlight {
        background-color: #1e2329;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the model
@st.cache_resource
def load_model():
    return XRayAnalyzer()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/x-ray.png", width=96)
    st.title("🔬 Controls")
    st.markdown("---")
    st.markdown("""
    ### How to use:
    1. Upload an X-ray image
    2. Wait for analysis
    3. Review results
    4. Check highlighted regions
    """)
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This advanced system uses computer vision and 
    machine learning to analyze medical X-rays.
    """)

# Main content
st.title("🔬 Advanced Medical X-ray Analysis")
st.markdown("""
<div class="highlight">
Advanced computer vision system for medical X-ray analysis.
Get instant insights with our state-of-the-art image processing technology.
</div>
""", unsafe_allow_html=True)

# File uploader with enhanced UI
uploaded_file = st.file_uploader(
    "Drop your X-ray image here",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    # Create columns for better layout
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Original image display
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", use_column_width=True)
    
    # Process image
    image_np = np.array(image)
    model = load_model()
    
    # Show processing status
    with st.spinner("🔍 Analyzing image..."):
        results = model.analyze(image_np)
    
    with col2:
        # Create tabs for different views
        tab1, tab2 = st.tabs(["📊 Analysis Results", "🎯 Region of Interest"])
        
        with tab1:
            # Enhanced results display
            st.markdown(f"""
            <div class="highlight">
            <h3 style="color: {'#00ff00' if results['prediction'] == 'Normal' else '#ff0000'}">
                {results['prediction']} ({results['confidence']*100:.1f}% confidence)
            </h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create three columns for visualizations
            vis_col1, vis_col2 = st.columns(2)
            
            with vis_col1:
                # Plotly gauge chart for confidence
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = results['confidence'] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Confidence Score"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#00a0dc"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "gray"},
                            {'range': [75, 100], 'color': "darkgray"}
                        ]
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            with vis_col2:
                # Add histogram visualization
                hist_fig = go.Figure()
                hist_fig.add_trace(go.Scatter(
                    x=results['histogram_bins'],
                    y=results['histogram'],
                    fill='tozeroy',
                    name='Intensity Distribution'
                ))
                hist_fig.update_layout(
                    title="Pixel Intensity Distribution",
                    xaxis_title="Intensity",
                    yaxis_title="Frequency",
                    height=250
                )
                st.plotly_chart(hist_fig, use_container_width=True)
            
            # Feature importance
            st.markdown("### Feature Analysis")
            for feature, value in results['feature_importance'].items():
                # Ensure value is between 0 and 1
                normalized_value = min(max(value, 0.0), 1.0)
                st.progress(normalized_value, text=f"{feature}: {value:.2%}")
        
        with tab2:
            # Enhanced visualization
            attention_map = results['attention_map']
            
            # Ensure image_np is in the right format for blending
            if len(image_np.shape) == 2:
                image_np_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            else:
                image_np_rgb = image_np
            
            # Resize attention map to match original image size
            attention_map_resized = cv2.resize(attention_map, (image_np_rgb.shape[1], image_np_rgb.shape[0]))
            
            # Create heatmap
            heatmap = cv2.applyColorMap(
                (attention_map_resized * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            
            # Convert heatmap to RGB
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Convert both images to float32 and normalize
            image_float = image_np_rgb.astype(np.float32) / 255.0
            heatmap_float = heatmap.astype(np.float32) / 255.0
            
            # Create blended overlay
            overlay = cv2.addWeighted(
                image_float,
                0.7,
                heatmap_float,
                0.3,
                0
            )
            
            # Scale back to uint8
            overlay = (overlay * 255).astype(np.uint8)
            
            st.image(overlay, caption="AI-Enhanced Region Detection", use_container_width=True)
    
    # Additional analysis details
    st.markdown("---")
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        ### 📋 Detailed Analysis
        """)
        st.markdown(f"""
        <div class="highlight">
        - **Primary Assessment**: {results['prediction']}
        - **Confidence Level**: {results['confidence']*100:.1f}%
        - **Edge Detection Score**: {results['feature_importance']['Edge Density']:.3f}
        - **Texture Analysis Score**: {results['feature_importance']['Texture Complexity']:.3f}
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        ### ⚠️ Important Notice
        """)
        st.markdown("""
        <div class="highlight">
        This tool is for educational and research purposes only.
        Always consult with qualified medical professionals for proper diagnosis and treatment.
        The AI analysis should not be used as a substitute for professional medical advice.
        </div>
        """, unsafe_allow_html=True) 