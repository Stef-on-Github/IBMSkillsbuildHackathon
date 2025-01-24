from flask import Flask, render_template, Response, request, jsonify
from ibm_watson import TextToSpeechV1, SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import os, logging, time, cv2
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
import numpy as np
import math
import threading
import webbrowser

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load API keys from environment variables
TEXT_TO_SPEECH_API_KEY = os.getenv('TEXT_TO_SPEECH_API_KEY')
TEXT_TO_SPEECH_URL = os.getenv('TEXT_TO_SPEECH_URL')
SPEECH_TO_TEXT_API_KEY = os.getenv('SPEECH_TO_TEXT_API_KEY')
SPEECH_TO_TEXT_URL = os.getenv('SPEECH_TO_TEXT_URL')

if not all([TEXT_TO_SPEECH_API_KEY, TEXT_TO_SPEECH_URL, SPEECH_TO_TEXT_API_KEY, SPEECH_TO_TEXT_URL]):
    logger.error("One or more IBM Watson API credentials are missing.")
    exit(1)

# Setup Text to Speech service
tts_authenticator = IAMAuthenticator(TEXT_TO_SPEECH_API_KEY)
text_to_speech = TextToSpeechV1(authenticator=tts_authenticator)
text_to_speech.set_service_url(TEXT_TO_SPEECH_URL)

# Setup Speech to Text service
stt_authenticator = IAMAuthenticator(SPEECH_TO_TEXT_API_KEY)
speech_to_text = SpeechToTextV1(authenticator=stt_authenticator)
speech_to_text.set_service_url(SPEECH_TO_TEXT_URL)

# Load all models
try:
    gesture_model = load_model("models/gesture_recognition_model.h5")
    alphabet_model = load_model("models/alphabet_model.h5")
    number_model = load_model("models/number_gesture_model.h5")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    exit(1)

detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

# Define labels for each model
models = {
    'gesture': {'model': gesture_model, 'labels': ['Hello', 'deaf', 'I love you', 'Yes', 'No', 'Please', 'Thank you', 'bathroom', 'Drink', 'Food/eat', 'Thirsty', 'Say', 'Know', 'No Idea', 'Forget', 'There is a link']},
    'alphabet': {'model': alphabet_model, 'labels': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'},
    'number': {'model': number_model, 'labels': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']}
}

current_model = 'gesture'  # Default model

cap = cv2.VideoCapture(0)

frame_counter = 0
skip_frames = 2  # Adjust this value based on your needs

def gen_video_feed():
    global frame_counter, current_model
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            continue
        frame_counter += 1
        if frame_counter % skip_frames != 0:
            ret, buffer = cv2.imencode('.jpg', img)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            continue
        
        hands, _ = detector.findHands(img)
        detected_letter = ''
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            y1 = max(0, y - offset)
            y2 = min(img.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(img.shape[1], x + w + offset)
            imgCrop = img[y1:y2, x1:x2]
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
            
            imgWhite = imgWhite.reshape(1, imgSize, imgSize, 3)
            selected_model = models[current_model]['model']
            prediction = selected_model.predict(imgWhite)
            index = np.argmax(prediction)
            detected_letter = models[current_model]['labels'][index]

            cv2.rectangle(img, (x - offset, y - offset-50),
                          (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
            cv2.putText(img, detected_letter, (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(img, (x-offset, y-offset),
                          (x + w+offset, y + h+offset), (255, 0, 255), 4)
            
            # Send detected letter to client
            app.config['DETECTED_LETTER'] = detected_letter
        
        ret, buffer = cv2.imencode('.jpg', img)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/synthesize', methods=['POST'])
def synthesize():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'Text is required for synthesis.'}), 400
        response = text_to_speech.synthesize(
            text=text,
            voice='en-US_AllisonV3Voice',
            accept='audio/wav'
        ).get_result()
        audio = response.content
        return app.response_class(audio, mimetype='audio/wav')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided.'}), 400
    audio_file = request.files['audio']
    file_extension = audio_file.filename.split('.')[-1]
    if file_extension == 'wav':
        content_type = 'audio/wav'
    elif file_extension == 'mp3':
        content_type = 'audio/mp3'
    else:
        return jsonify({'error': 'Unsupported audio format. Please upload a .wav or .mp3 file.'}), 400
    try:
        response = speech_to_text.recognize(
            audio=audio_file,
            content_type=content_type,
            model='en-US_BroadbandModel'
        ).get_result()
        if response['results']:
            transcribed_text = response['results'][0]['alternatives'][0]['transcript']
            return jsonify({'transcription': transcribed_text})
        return jsonify({'error': 'No speech detected in the audio.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/transcribe-realtime', methods=['POST'])
def transcribe_realtime():
    try:
        data = request.get_json()
        text = data.get('audio', '')
        if not text:
            return jsonify({'error': 'Audio data is required for transcription.'}), 400
        # For simplicity, we are returning the same text as the transcription
        # In a real-world scenario, you might want to process this text further
        return jsonify({'transcription': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    return Response(gen_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_letter')
def get_letter():
    def event_stream():
        last_detected_letter = ''
        while True:
            detected_letter = app.config.get('DETECTED_LETTER', '')
            if detected_letter and detected_letter != last_detected_letter:
                yield f"data: {detected_letter}\n\n"
                last_detected_letter = detected_letter
            time.sleep(0.1)  # Adjust the sleep duration as needed
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/set_model', methods=['POST'])
def set_model():
    global current_model
    selected_model = request.form.get('model')
    if selected_model in models:
        current_model = selected_model
        print(f"Switched to {selected_model} model")
        return '', 204
    else:
        print(f"Invalid model selection: {selected_model}")
        return jsonify({'error': 'Invalid model'}), 400

def print_link():
    url = "http://127.0.0.1:5000/"
    print(f"Application running at: {url}")
    webbrowser.open_new(url)

if __name__ == "__main__":
    threading.Thread(target=print_link).start()
    app.run(debug=True, use_reloader=False)