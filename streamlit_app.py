#!/usr/bin/env python3
"""
Streamlit Cloud Entry Point
This file is used by Streamlit Cloud to run the fraud detection dashboard
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the dashboard
from dashboard import main

if __name__ == "__main__":
    main() 