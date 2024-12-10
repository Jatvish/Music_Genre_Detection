import os
import librosa
import json
import numpy as np
import math

DATASET_PATH = r"C:\Users\akaas\OneDrive\Desktop\GTZAN\genres_original"  # Path to the GTZAN dataset or your audio folder
JSON_PATH = "mfcc_dataset.json"  # Path to save the MFCC dataset as a JSON file

SAMPLE_RATE = 22050
DURATION = 30  # All files will be cut or padded to 30 seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
NUM_MFCC = 13  # Number of MFCCs to extract

def save_mfcc(dataset_path, json_path, num_mfcc=NUM_MFCC, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from the dataset and saves them in a json file along with labels."""

    data = {
        "mfcc": [],  # MFCC features
        "labels": [],  # Genre labels
        "mapping": []  # To map label indices to genre names
    }

    # Loop through all genre sub-folders
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # Ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # Save genre label (sub-folder name) in the mapping
            genre_label = dirpath.split("/")[-1]
            data["mapping"].append(genre_label)
            print(f"Processing genre: {genre_label}")

            # Process files for each genre
            for f in filenames:
                
                # Load audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

                # Process the file in chunks (segments) to ensure consistent length for all samples
                num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
                expected_num_mfcc_vectors = math.ceil(num_samples_per_segment / hop_length)

                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                sr=sr, n_mfcc=num_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)

                    mfcc = mfcc.T  # Transpose to have time steps as rows

                    # Store MFCC if it has the expected length
                    if len(mfcc) == expected_num_mfcc_vectors:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)  # Label index (i - 1 because first dir is the dataset root)

    # Save the MFCCs and labels in a json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH)
