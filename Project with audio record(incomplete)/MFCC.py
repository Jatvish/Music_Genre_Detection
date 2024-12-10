import os
import librosa
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow import keras
import threading

# Paths
MODEL_PATH = r"C:\Users\jatvi\OneDrive\Desktop\5th Sem\SE\SE PROJECT\cnn_improved.keras"
BACKGROUND_IMAGE_PATH = r"C:\Users\jatvi\OneDrive\Desktop\5th Sem\SE\SE PROJECT\background.jpeg"
GENRES = ["Blues", "Classical", "Country", "Disco", "HipHop", "Jazz", "Metal", "Pop", "Reggae", "Rock"]

# Constants for audio processing
SAMPLE_RATE = 22050
DURATION = 30  # Duration of each audio track
RECORDED_FILE_PATH = "recorded_audio.wav"

# Load the trained model
model = keras.models.load_model(MODEL_PATH)

# Function to record audio from the microphone
def record_audio(duration=DURATION):
    try:
        print("Recording audio...")
        audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()  # Wait until the recording is finished
        write(RECORDED_FILE_PATH, SAMPLE_RATE, audio_data)  # Save as WAV file
        print(f"Recording saved to {RECORDED_FILE_PATH}")
        return RECORDED_FILE_PATH
    except Exception as e:
        print(f"Error recording audio: {e}")
        messagebox.showerror("Recording Error", f"Error recording audio: {e}")
        return None

# Function to preprocess audio for prediction
def preprocess_audio(file_path):
    try:
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(signal, sr=sr, n_mfcc=13)
        mfcc = mfcc.T
        mfcc = mfcc[np.newaxis, ..., np.newaxis]
        return mfcc
    except Exception as e:
        print(f"Error processing audio: {e}")
        messagebox.showerror("Processing Error", f"Error processing audio: {e}")
        return None

# Function to predict genre based on audio file
def predict_genre(file_path):
    if model is None:
        messagebox.showerror("Error", "Model is not loaded.")
        return
    mfcc = preprocess_audio(file_path)
    if mfcc is not None:
        prediction = model.predict(mfcc)
        predicted_index = np.argmax(prediction)
        predicted_genre = GENRES[predicted_index]
        result_label.config(text=f"Detected Genre: {predicted_genre}")
    else:
        result_label.config(text="Please insert a valid file.")

# Function to handle recording in a separate thread
def record_and_predict():
    recorded_file = record_audio()
    if recorded_file:
        predict_genre(recorded_file)

# Function to open file dialog and predict genre from selected file
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])
    if file_path:
        predict_genre(file_path)
    else:
        result_label.config(text="Please insert a valid file.")

# GUI setup with Tkinter
root = tk.Tk()
root.title("Music Genre Classifier")

# Full screen setup
root.attributes("-fullscreen", True)

# Load background image
background_image = Image.open(BACKGROUND_IMAGE_PATH)
background_image = background_image.resize((1920, 1080), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(background_image)
bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Add UI elements
title_label = tk.Label(root, text="Music Genre Classifier", font=("Helvetica", 40, "bold"), bg=root.cget("bg"), fg="#333")
title_label.pack(pady=30)

# Record and classify button
record_button = tk.Button(root, text="Record Audio & Predict", font=("Helvetica", 24), command=lambda: threading.Thread(target=record_and_predict).start(), bg="#2980b9", fg="white", bd=0, relief="flat", height=2, width=20)
record_button.pack(pady=20)

# Select audio file button
predict_button = tk.Button(root, text="Select Audio File", font=("Helvetica", 24), command=open_file, bg="#2980b9", fg="white", bd=0, relief="flat", height=2, width=20)
predict_button.pack(pady=20)

# Result label
result_label = tk.Label(root, text="Detected Genre: ", font=("Helvetica", 28, "italic"), bg=root.cget("bg"), fg="#333")
result_label.pack(pady=30)

# Run the GUI loop
root.mainloop()
