import os
import numpy as np
import librosa
import uuid
import traceback
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import random

app = Flask(__name__)
CORS(app)

# Constants
MODEL_PATH = "gender_model.h5"
N_MFCC = 128
MAX_LEN = 128
UPLOAD_FOLDER = "/tmp"

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def check_ffmpeg():
    import subprocess
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

HAS_FFMPEG = check_ffmpeg()
if not HAS_FFMPEG:
    print("WARNING: ffmpeg not found. Audio processing might fail for non-WAV files.")

# Global Model Variable
model = None

def load_gender_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    else:
        print("Model file not found. Using mock prediction.")
        model = None

def extract_features_from_audio(file_path):
    try:
        # Load audio using librosa
        # librosa.load can handle many formats if ffmpeg is present
        # If it fails, we might need to convert it first
        audio, sample_rate = librosa.load(file_path, sr=None)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
        
        # Pad or truncate to MAX_LEN
        if mfccs.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :MAX_LEN]
            
        return mfccs.T # Shape: (128, 128)
    except Exception as e:
        print(f"Error processing audio with librosa: {e}")
        if not HAS_FFMPEG:
            print("Fallback skipped: FFMPEG is not installed.")
            return None
        try:
             # Fallback: try pydub to convert to wav if it's webm
             from pydub import AudioSegment
             audio_seg = AudioSegment.from_file(file_path)
             wav_path = file_path + ".wav"
             audio_seg.export(wav_path, format="wav")
             
             audio, sample_rate = librosa.load(wav_path, sr=None)
             os.remove(wav_path) # Clean up temp wav

             mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
             if mfccs.shape[1] < MAX_LEN:
                pad_width = MAX_LEN - mfccs.shape[1]
                mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
             else:
                mfccs = mfccs[:, :MAX_LEN]
             return mfccs.T.reshape(MAX_LEN, N_MFCC)
        except Exception as e2:
            print(f"Fallback processing failed: {e2}")
            traceback.print_exc()
            return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'message': 'Backend is reachable'})

@app.route('/predict', methods=['POST'])
def predict():
    print("Received prediction request")
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Generate a unique filename
    # MediaRecorder usually sends webm, so we label it accordingly
    filename = f"{uuid.uuid4()}.webm"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        audio_file.save(file_path)
        
        # Prediction Logic
        if model:
            features = extract_features_from_audio(file_path)
            if features is not None:
                prediction = model.predict(np.expand_dims(features, axis=0))[0][0]
                # > 0.5 is Female (1), < 0.5 is Male (0)
                # Note: Adjust threshold/label based on specific training data labeling
                gender = "Female" if prediction > 0.5 else "Male"
                confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)
            else:
                 # If feature extraction fails and we don't have FFmpeg, we can Fallback to Mock 
                 # or return error. Let's return a specific error if FFmpeg is missing.
                 if not HAS_FFMPEG:
                     return jsonify({
                         'error': 'FFMPEG missing', 
                         'message': 'Server requires FFMPEG to process browser audio. Please install FFMPEG.'
                     }), 500
                 return jsonify({'error': 'Could not process audio features'}), 500
        else:
            # MOCK MODE for Demo without trained model
            print("Using Mock Prediction")
            is_female = random.choice([True, False])
            gender = "Female" if is_female else "Male"
            confidence = random.uniform(0.75, 0.98)

        return jsonify({
            'gender': gender,
            'confidence': f"{confidence * 100:.2f}%"
        })
        
    except Exception as e:
        print(f"EXCEPTION in /predict: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f"Server Error: {str(e)}"}), 500
    finally:
        # Clean up the file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error removing temp file: {e}")

if __name__ == '__main__':
    load_gender_model()
    # Provide a clear message about the server URL
    print("Starting Flask Server...")
    print("Access the app at http://127.0.0.1:5001")
    app.run(port=5001, debug=True)
