import streamlit as st
import numpy as np
import time

# Simple Streamlit app
st.title("Simple X-ray Analyzer Demo")
st.write("This is a minimal demo to verify Streamlit is working")

# Add a simple button
if st.button("Click me"):
    with st.spinner("Processing..."):
        time.sleep(2)  # Simulate work
    st.success("Done!")
    
    # Display a simple chart
    chart_data = np.random.randn(20, 3)
    st.line_chart(chart_data)
    
    st.write("In the full app, you would see X-ray analysis here.")
    
st.write("If you can see this and interact with the button, Streamlit is working correctly!") 