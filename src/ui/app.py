"""
Streamlit app for X-ray analysis (Demo version)
"""

import streamlit as st
import cv2
import numpy as np
import os
from pathlib import Path

class XRayAnalyzerDemoApp:
    def __init__(self):
        """Initialize the Streamlit demo app"""
        # Create upload directory if it doesn't exist
        os.makedirs('uploads', exist_ok=True)

    def run(self):
        """Run the Streamlit app"""
        st.title("Medical X-ray Analyzer (Demo)")
        st.write("""
        Upload an X-ray image to analyze it for potential medical conditions.
        
        Note: This is a demo version that shows the UI functionality only.
        The actual model prediction functionality is not available in this demo.
        """)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an X-ray image",
            type=['.jpg', '.jpeg', '.png']
        )
        
        if uploaded_file is not None:
            # Check file size
            if uploaded_file.size > 5 * 1024 * 1024:  # 5MB
                st.error("File size too large. Please upload a smaller file.")
                return
            
            # Display original image
            col1, col2 = st.columns(2)
            
            # Save uploaded file temporarily
            save_path = Path('uploads') / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Read and display image
            image = cv2.imread(str(save_path))
            if image is None:
                st.error("Error reading the image. Please try another file.")
                return
                
            with col1:
                st.subheader("Original X-ray")
                st.image(uploaded_file, use_column_width=True)
            
            # Process button
            if st.button("Analyze X-ray"):
                with st.spinner("Analyzing image..."):
                    # Simulate processing delay
                    import time
                    time.sleep(2)
                    
                    # Display results (demo visualization)
                    with col2:
                        st.subheader("Analysis Results (Demo)")
                        
                        # Display demo prediction
                        st.write("### Diagnosis")
                        condition_detected = True  # Demo value
                        
                        if condition_detected:
                            st.error("⚠️ Potential medical condition detected")
                            st.write(f"Confidence: {0.78:.2%}")
                        else:
                            st.success("✅ No significant issues detected")
                            st.write(f"Confidence: {0.92:.2%}")
                        
                        # Display demo heatmap
                        st.write("### Region Analysis")
                        
                        # Create a simulated heatmap for demo
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
                        
                        # Create a heatmap-like visualization
                        heatmap = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
                        # Add some transparency
                        alpha = 0.6
                        overlay = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
                        
                        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_column_width=True)
                        st.info("The heatmap highlights regions that would influence the model's decision.")
                        
                        st.write("Note: In the full version, this would be a real Grad-CAM visualization.")
            
            # Clean up
            os.remove(save_path)

def main():
    app = XRayAnalyzerDemoApp()
    app.run()

if __name__ == "__main__":
    main() 