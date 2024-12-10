import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
from sklearn.model_selection import train_test_split

# Paths and configurations
MODEL_PATH = os.path.join(r"C:\Users\jatvi\OneDrive\Desktop\5th Sem\SE\SE PROJECT", "cnn_improved.keras")  # Adjust path
FEATURE_PATH = "mfcc.json"  # Path to MFCC dataset
RECORDED_FILE_PATH = "recorded_audio.wav"  # Path to save recorded audio
GENRES = ["Blues", "Classic", "Country", "Disco", "Hippop", "Jazz", "Metal", "Pop", "Reggae", "Rock"]

def load_data(data_path):
    """Loads training dataset from JSON file."""
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y

def prepare_datasets(test_size, validation_size):
    """Loads data and splits it into train, validation, and test sets."""
    X, y = load_data(FEATURE_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # Add an axis to input sets for CNN
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape):
    """Generates a CNN model with additional dropout layers for robustness."""
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(64, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.GlobalAveragePooling2D())

    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.4))

    model.add(keras.layers.Dense(10, activation='softmax'))

    return model

def record_audio(duration=30):
    """Records audio from the microphone and saves it as a .wav file."""
    print("Recording...")
    audio_data = sd.rec(int(duration * 22050), samplerate=22050, channels=1)
    sd.wait()  # Wait until the recording is finished
    write(RECORDED_FILE_PATH, 22050, audio_data)  # Save as WAV file
    print(f"Recording saved to {RECORDED_FILE_PATH}")
    return RECORDED_FILE_PATH

def preprocess_audio(file_path, n_mfcc=40, duration=30):
    """Preprocesses input audio file to extract MFCC features without augmentation for testing."""
    try:
        signal, sr = librosa.load(file_path, sr=22050, duration=duration)

        # Extract MFCC features from the audio signal
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        mfcc = mfcc.T
        mfcc = mfcc[np.newaxis, ..., np.newaxis]  # Add batch dimension and channel dimension

        return mfcc

    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def predict_genre(file_path, model, num_predictions=5):
    """Predicts the genre of an audio file multiple times and averages results for consistency."""
    predictions = []
    for _ in range(num_predictions):
        mfcc = preprocess_audio(file_path)
        if mfcc is not None:
            prediction = model.predict(mfcc)
            predictions.append(prediction)

    if predictions:
        avg_prediction = np.mean(predictions, axis=0)  # Average predictions
        predicted_index = np.argmax(avg_prediction, axis=1)[0]
        predicted_genre = GENRES[predicted_index]
        print(f"Predicted genre: {predicted_genre}")
    else:
        print("Unable to predict the genre.")

if __name__ == "__main__":
    # Load dataset and prepare data for training
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape)

    optimiser = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    _ = model.fit(X_train, y_train,
                  validation_data=(X_validation, y_validation),
                  batch_size=32,
                  epochs=100,
                  callbacks=[early_stopping])

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    # Save the trained model
    model.save(MODEL_PATH)

    # Load model and make predictions on recorded audio
    model = keras.models.load_model(MODEL_PATH)
    choice = input("Type 'record' to record audio or 'file' to provide an audio file path: ").strip().lower()

    if choice == "record":
        audio_file_path = record_audio()
    else:
        audio_file_path = input("Enter the path to the audio file: ")

    # Predict genre from the audio file
    predict_genre(audio_file_path, model)
