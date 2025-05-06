#!/usr/bin/env python
"""
Script to start the X-ray analyzer Streamlit app
"""

import streamlit.web.bootstrap as bootstrap
import os
import sys

def run_app():
    """Run the Streamlit app"""
    # Get directory of this script
    dirname = os.path.dirname(__file__)
    
    # Add the current directory to the path
    sys.path.insert(0, dirname)
    
    # Run the Streamlit app
    bootstrap.run(os.path.join(dirname, "src", "ui", "app.py"), "", [], [])

if __name__ == "__main__":
    run_app() 