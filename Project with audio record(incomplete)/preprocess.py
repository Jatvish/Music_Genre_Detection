import json
import os
import math
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
from config import DATASET_PATH, FEATURE_PATH, MAPPING_PATH, SAMPLE_RATE, SAMPLES_PER_TRACK, NUM_SEGMENTS, TRACK_DURATION_SECONDS

RECORDED_FILE_PATH = "recorded_audio.wav"  # Path to save recorded audio

# Function to record audio from the microphone and save it as a .wav file
def record_audio(duration=TRACK_DURATION_SECONDS):
    print("Recording...")
    audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()  # Wait until the recording is finished
    write(RECORDED_FILE_PATH, SAMPLE_RATE, audio_data)  # Save as WAV file
    print(f"Recording saved to {RECORDED_FILE_PATH}")
    return RECORDED_FILE_PATH

# Function to save MFCCs from the dataset or a single audio file
def save_mfcc(dataset_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5, single_file=None):
    """
    Extracts MFCCs from the dataset or a single audio file and saves them in a JSON file with genre labels.

    :param dataset_path (str): Path to dataset
    :param num_mfcc (int): Number of MFCC coefficients to extract
    :param n_fft (int): Interval to apply FFT
    :param hop_length (int): Sliding window for FFT
    :param num_segments (int): Number of segments to divide audio into
    :param single_file (str): Path to a single audio file (optional)
    """
    
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    if single_file:
        # Process a single audio file (e.g., recorded audio)
        print(f"Processing file: {single_file}")
        process_file(single_file, data, num_mfcc, n_fft, hop_length, num_segments, samples_per_segment, num_mfcc_vectors_per_segment)
    
    else:
        # Loop through all genre folders in the dataset
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
            if dirpath is not dataset_path:
                genre_label = os.path.basename(dirpath)
                data["mapping"].append(genre_label)
                print(f"\nProcessing genre: {genre_label}")

                for f in filenames:
                    file_path = os.path.join(dirpath, f)
                    process_file(file_path, data, num_mfcc, n_fft, hop_length, num_segments, samples_per_segment, num_mfcc_vectors_per_segment, i-1)
    
    # Save MFCCs to JSON file
    with open(FEATURE_PATH, "w") as fp:
        json.dump(data, fp, indent=2)

def process_file(file_path, data, num_mfcc, n_fft, hop_length, num_segments, samples_per_segment, num_mfcc_vectors_per_segment, label=None):
    """
    Processes a single audio file, extracts MFCCs, and appends to data dictionary.

    :param file_path (str): Path to the audio file
    :param data (dict): Dictionary to store MFCCs and labels
    :param samples_per_segment (int): Number of samples per segment
    :param num_mfcc_vectors_per_segment (int): Expected number of MFCC vectors per segment
    :param label (int): Genre label for dataset processing; None for single file
    """
    try:
        signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE, duration=TRACK_DURATION_SECONDS)

        # Process all segments of audio file
        for d in range(num_segments):
            start = d * samples_per_segment
            finish = start + samples_per_segment

            mfcc = librosa.feature.mfcc(signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc = mfcc.T

            if len(mfcc) == num_mfcc_vectors_per_segment:
                data["mfcc"].append(mfcc.tolist())
                data["labels"].append(label if label is not None else -1)  # Assign a placeholder label if none provided
    except Exception as e:
        print(f"Could not process {file_path}: {e}")

if __name__ == "__main__":
    # To process the whole dataset
    save_mfcc(DATASET_PATH, num_segments=NUM_SEGMENTS)

    # To record audio and process it for MFCC
    recorded_file = record_audio()
    save_mfcc(DATASET_PATH, single_file=recorded_file, num_segments=NUM_SEGMENTS)
