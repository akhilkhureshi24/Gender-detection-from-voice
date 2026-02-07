import os
import numpy as np
import librosa
import glob
from sklearn.model_selection import train_test_split
from model import build_model
import tensorflow as tf

# Constants
DATA_PATH = "dataset"  # Folder structure: dataset/male/*.wav, dataset/female/*.wav
N_MFCC = 128
MAX_LEN = 128  # Time steps

def extract_features(file_path, augment=False):
    """
    Extracts 128 MFCC features from an audio file with optional augmentation.
    """
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        
        if augment:
            # Simple augmentation: pitch shift
            pitch_step = np.random.uniform(-2, 2)
            audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_step)
            # Add noise
            noise = np.random.randn(len(audio))
            audio = audio + 0.005 * noise

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
        
        # Pad or truncate to MAX_LEN
        if mfccs.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :MAX_LEN]
            
        return mfccs.T  # Shape: (128, 128)
    except Exception as e:
        print(f"Error parsing file: {file_path} - {e}")
        return None

def load_data():
    X = []
    y = []
    genders = {'male': 0, 'female': 1}
    
    for gender, label in genders.items():
        path = os.path.join(DATA_PATH, gender, "*.wav")
        files = glob.glob(path)
        print(f"Loading {len(files)} {gender} files...")
        
        for file in files:
            # Original
            features = extract_features(file)
            if features is not None:
                X.append(features)
                y.append(label)
            
            # Augmented (to increase dataset size and robustness)
            features_aug = extract_features(file, augment=True)
            if features_aug is not None:
                X.append(features_aug)
                y.append(label)
                
    return np.array(X), np.array(y)

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"Dataset directory '{DATA_PATH}' not found. Generating dummy data for demonstration...")
        X = np.random.rand(20, MAX_LEN, N_MFCC)
        y = np.random.randint(0, 2, size=(20,))
    else:
        X, y = load_data()
        if len(X) == 0:
             print("No audio files found. Using dummy data.")
             X = np.random.rand(20, MAX_LEN, N_MFCC)
             y = np.random.randint(0, 2, size=(20,))
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model((MAX_LEN, N_MFCC))
    
    print("Starting training...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    model.save("gender_model.h5")
    print("Model saved to gender_model.h5")
