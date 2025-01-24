import flask
from ibm_watson import TextToSpeechV1
import ibm_cloud_sdk_core
import os
import logging
import cv2
import cvzone
import tensorflow as tf
import numpy as np
import importlib.metadata

def print_library_versions():
    try:
        print(f"Flask version: {importlib.metadata.version('flask')}")
    except Exception as e:
        print(f"Error getting Flask version: {e}")
    try:
        print(f"IBM Watson Text to Speech version: N/A (version attribute not available)")
    except Exception as e:
        print(f"Error getting IBM Watson Text to Speech version: {e}")
    try:
        print(f"IBM Cloud SDK Core version: {importlib.metadata.version('ibm-cloud-sdk-core')}")
    except Exception as e:
        print(f"Error getting IBM Cloud SDK Core version: {e}")
    try:
        print(f"OpenCV version: {cv2.__version__}")
    except Exception as e:
        print(f"Error getting OpenCV version: {e}")
    try:
        print(f"Cvzone version: N/A (version attribute not available)")
    except Exception as e:
        print(f"Error getting Cvzone version: {e}")
    try:
        print(f"TensorFlow version: {tf.__version__}")
    except Exception as e:
        print(f"Error getting TensorFlow version: {e}")
    try:
        print(f"Numpy version: {np.__version__}")
    except Exception as e:
        print(f"Error getting Numpy version: {e}")
    try:
        print("OS, Logging, Math, Threading, and Webbrowser are standard Python libraries and do not have specific versions.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print_library_versions()
