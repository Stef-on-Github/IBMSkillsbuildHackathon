# Empowering Accessibility: A System for Enhanced Computer Interaction for People with Disabilities

## Team Name
**Cirrus Creators**

## Project Title
**Empowering Accessibility: A System for Enhanced Computer Interaction for People with Disabilities**

## Problem Statement
Individuals with physical disabilities often face significant barriers to accessing and interacting with computers. This project aims to develop an innovative system that empowers these individuals by providing alternative input and output methods.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributors](#contributors)
- [License](#license)

## Introduction
The **Empowering Accessibility** project is designed to enhance the interaction experience for people with disabilities. It integrates multiple technologies to provide alternative ways of interacting with computers, including sign language detection, text-to-speech, speech-to-text, and eye-controlled mouse cursor movement.

## Features
- **Sign Language Detection**: Real-time detection of sign language gestures using machine learning models.
- **Text to Speech Conversion**: Convert written text into spoken words using IBM Watson Text to Speech API.
- **Speech to Text Transcription**: Convert spoken words into written text using IBM Watson Speech to Text API.
- **Eye-Controlled Mouse Cursor**: Control the computer mouse cursor using eye movements (implemented in `cursor.py`).
- **Model Switching**: Ability to switch between different sign language recognition models (gesture, alphabet, numbers).

## Technologies Used
- **Python**: Programming language used for backend development.
- **Flask**: Web framework for serving HTML pages and handling HTTP requests.
- **IBM Watson**: APIs for text-to-speech and speech-to-text conversion.
- **TensorFlow/Keras**: Machine learning framework for loading and using pre-trained models.
- **OpenCV**: Library for real-time image processing and hand tracking.
- **HTML/CSS/JavaScript**: Frontend technologies for building the user interface.
- **EventSource**: For real-time updates from the server.
- **Web Speech API**: For voice recognition and synthesis.

## Installation
### Prerequisites
Make sure you have the following installed:
- Python 3.9
- pip (Python package installer)
- Virtual Environment (optional but recommended)

### Steps
1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <project-directory>
    ```

2. Create and activate a virtual environment (optional):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables for IBM Watson API keys:
    ```bash
    export TEXT_TO_SPEECH_API_KEY='your_text_to_speech_api_key'
    export TEXT_TO_SPEECH_URL='your_text_to_speech_url'
    export SPEECH_TO_TEXT_API_KEY='your_speech_to_text_api_key'
    export SPEECH_TO_TEXT_URL='your_speech_to_text_url'
    ```

5. Run the Flask application:
    ```bash
    python app.py
    ```

6. Open your web browser and navigate to `http://127.0.0.1:5000/`.

## Usage
- **Sign Language Detection**: Use the webcam to detect sign language gestures and convert them into text.
- **Text to Speech Conversion**: Enter text in the provided textarea and click "Convert to Speech" to hear it.
- **Speech to Text Transcription**: Upload an audio file or start recording to transcribe spoken words into text.
- **Eye-Controlled Mouse Cursor**: Use the `cursor.py` script to control the mouse cursor with your eyes.

## Project Structure
```
project-directory/
│
├── app.py                    # Main Flask application
├── cursor.py                 # Script for controlling mouse cursor with eyes
├── static/
│   ├── styles.css            # CSS stylesheet
│   └── favicon.png           # Favicon for the website
├── templates/
│   └── index.html            # HTML template for the main page
├── requirements.txt          # List of Python dependencies
└── README.md                 # This README file
```

## Contributors
- **[Stephen Mhetre]**: Team Member 01
- **[Prathamesh Kudale]**: Team Member 02
- **[Prof. Vaibhav Suryawanshi]**: Project Guide
- College Name - **[Nutan Institutes (NMVPM) ,Pune]** 

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
