import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, BatchNormalization, Flatten, Input

def build_model(input_shape):
    """
    Builds a hybrid CNN-LSTM model for gender prediction.
    CNN extracts local spectral features, LSTM models temporal dependencies.
    """
    model = Sequential()
    
    model.add(Input(shape=input_shape))
    
    # CNN Layer 1
    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    
    # CNN Layer 2
    model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    
    # LSTM Layers
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))
    
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))
    
    # Dense Layers
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    
    # Output Layer
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Example input shape: (Time Steps, Features) = (128, 128)
    model = build_model((128, 128))
    model.summary()
