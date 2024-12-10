import tkinter as tk
from tkinter import filedialog, messagebox
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import librosa
from sklearn.model_selection import train_test_split
import threading

MODEL_PATH = os.path.join(r"C:\Users\jatvi\OneDrive\Desktop\5th Sem\SE\SE PROJECT", "cnn_improved.keras")
FEATURE_PATH = "mfcc.json"
GENRES = ["Blues", "Classic", "Country", "Disco", "Hippop", "Jazz", "Metal", "Pop", "Reggae", "Rock"]

model = None  # Placeholder for model
train_accuracy = 0  # Placeholder for accuracy


def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y


def prepare_datasets(test_size, validation_size):
    X, y = load_data(FEATURE_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    model = keras.Sequential([
        keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (2, 2), activation='relu'),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model


def train_model_thread():
    global model, train_accuracy
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape)
    optimiser = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    progress_label.config(text="Training model... Please wait.")
    root.update_idletasks()
    
    # Train model with EarlyStopping
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=10, callbacks=[early_stopping])
    train_accuracy = history.history['accuracy'][-1] * 100

    # Evaluate the model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    progress_label.config(text=f"Model trained successfully! Test Accuracy: {test_acc * 100:.2f}%")
    model.save(MODEL_PATH)
    predict_button.config(state="normal")


def start_training():
    progress_label.config(text="Initializing training...")
    train_button.config(state="disabled")
    threading.Thread(target=train_model_thread).start()


def preprocess_audio(file_path, n_mfcc=40, duration=30):
    try:
        signal, sr = librosa.load(file_path, sr=None, duration=duration)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        mfcc = mfcc.T
        mfcc = mfcc[np.newaxis, ..., np.newaxis]
        return mfcc
    except Exception as e:
        messagebox.showerror("Error", f"Error processing file: {e}")
        return None


def predict_genre(file_path):
    if model is None:
        messagebox.showerror("Error", "Model is not trained or loaded.")
        return
    mfcc = preprocess_audio(file_path)
    if mfcc is not None:
        prediction = model.predict(mfcc)
        predicted_index = np.argmax(prediction)
        predicted_genre = GENRES[predicted_index]
        result_label.config(text=f"Detected Genre: {predicted_genre}")
    else:
        result_label.config(text="Please insert a valid file.")


def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if file_path:
        predict_genre(file_path)
    else:
        result_label.config(text="Please insert a valid file.")


# GUI setup
root = tk.Tk()
root.title("Music Genre Classifier")
root.geometry("450x300")

title_label = tk.Label(root, text="Music Genre Classifier", font=("Arial", 16))
title_label.pack(pady=10)

train_button = tk.Button(root, text="Train Model", command=start_training, font=("Arial", 12))
train_button.pack(pady=10)

progress_label = tk.Label(root, text="", font=("Arial", 12), fg="blue")
progress_label.pack(pady=10)

predict_button = tk.Button(root, text="Select Audio File", command=open_file, font=("Arial", 12), state="disabled")
predict_button.pack(pady=10)

result_label = tk.Label(root, text="Detected Genre: ", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
