import subprocess
import sys
import os

# Ensure correct protobuf version
try:
    import google.protobuf
    current_version = google.protobuf.__version__
    if current_version != '3.20.0':
        print(f"Current protobuf version is {current_version}, upgrading to 3.20.0")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "protobuf==3.20.0"])
except ImportError:
    print("protobuf is not installed, installing 3.20.0...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "protobuf==3.20.0"])

# Set the environment variable to use the pure Python implementation of protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st

# Streamlit App Content
st.title("Streamlit App Example")
st.write("This is a Streamlit app that installs or downgrades protobuf and sets the implementation to python.")
st.write("If protobuf isn't already at version 3.20.0, it will be automatically installed.")

# Add some content to the Streamlit app
st.write("You can now continue with your Streamlit app functionality here.")
