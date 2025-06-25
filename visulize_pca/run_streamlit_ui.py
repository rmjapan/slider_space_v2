#!/usr/bin/env python3
"""
Launcher script for the Streamlit PCA UI with PyTorch compatibility fixes.
This script helps avoid common Streamlit + PyTorch compatibility issues.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Set up environment variables to avoid Streamlit + PyTorch conflicts."""
    
    # Disable Streamlit file watcher to avoid PyTorch module inspection issues
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_THEME_BASE"] = "light"
    
    # Set PyTorch settings for better compatibility
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Disable some warnings that can interfere with Streamlit
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:torch"

def run_streamlit():
    """Run the Streamlit app with proper configuration."""
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    streamlit_app = script_dir / "pca_visualization_ui_enhanced.py"
    
    if not streamlit_app.exists():
        print(f"Error: Streamlit app not found at {streamlit_app}")
        sys.exit(1)
    
    # Set up environment
    setup_environment()
    
    # Run streamlit with specific configuration
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(streamlit_app),
        "--server.fileWatcherType", "none",
        "--browser.gatherUsageStats", "false",
        "--server.address", "0.0.0.0",
        "--server.port", "8501",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ]
    
    print("Starting Streamlit PCA UI...")
    print(f"App will be available at: http://localhost:8505")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStreamlit server stopped.")
    except Exception as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_streamlit() 