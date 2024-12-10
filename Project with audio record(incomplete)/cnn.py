import json
import numpy as np
import sounddevice as sd
import wavio  # Make sure to install the wavio package for saving .wav files
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras 
from config import MODEL_PATH, FEATURE_PATH
import os

MODEL_PATH = os.path.join(MODEL_PATH, "cnn.keras")

def load_data(data_path):
    """Loads training dataset from json file.
    
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y

def prepare_datasets(test_size, validation_size):
    """Loads data and splits it into train, validation and test sets.
    
    :param test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split
    :param validation_size (float): Value in [0, 1] indicating percentage of train set to allocate to validation split

    :return X_train (ndarray): Input training set
    :return X_validation (ndarray): Input validation set
    :return X_test (ndarray): Input test set
    :return y_train (ndarray): Target training set
    :return y_validation (ndarray): Target validation set
    :return y_test (ndarray): Target test set
    """
    # load data
    X, y = load_data(FEATURE_PATH)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # add an axis to input sets
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape):
    """Generates CNN model
    
    :param input_shape (tuple): Shape of input set
    :return model: CNN model
    """
    # build network topology
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model

def record_audio(filename, duration=5, fs=44100):
    """Records audio from the microphone and saves it as a .wav file.
    
    :param filename (str): Filename to save the recorded audio
    :param duration (int): Duration of the recording in seconds
    :param fs (int): Sampling frequency
    """
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
    sd.wait()  # Wait until recording is finished
    wavio.write(filename, audio, fs, sampwidth=2)
    print("Recording saved as:", filename)

def preprocess_audio(filename):
    """Preprocesses the audio file to be input for the model.
    
    :param filename (str): Path to the .wav file to be processed
    :return mfcc (ndarray): MFCC features extracted from the audio
    """
    # Load the audio file and extract features (this is a placeholder function)
    # You need to implement the actual MFCC extraction based on your requirements
    import librosa
    y, sr = librosa.load(filename, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.T  # Transpose to match expected input shape

def predict_genre(model, audio_file):
    """Predicts the genre of the audio file using the trained model.
    
    :param model: Trained CNN model
    :param audio_file (str): Path to the audio file
    :return genre (str): Predicted genre
    """
    mfcc = preprocess_audio(audio_file)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
    prediction = model.predict(mfcc)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class

if __name__ == "__main__":
    # get train, validation, test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape)

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0015)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # train model
    _ = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=50)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    # save model
    model.save(MODEL_PATH)

    # Record audio from the microphone and predict genre
    audio_filename = "recorded_audio.wav"
    record_audio(audio_filename, duration=5)  # Adjust the duration as needed
    predicted_genre = predict_genre(model, audio_filename)
    print('Predicted genre class:', predicted_genre)
