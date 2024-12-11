Requirements and libraries required: 

-python
-librosa
-tensorflow
-pillow (used in front end)
-numpy

// might have missed some libraries

Steps to implement the project:

1. Download all the files and the GTZAN dataset, which has all the audio files.
2. Install all required libraries
3. Have the python environment ready in terminal 	//  .\venv\Scripts\activate
4. run these in order: config.py, preprocess.py, usertest.py(optional), cnn.py(need to run all 50 epocs), MFCC.py
5. run the main code Front_test.py
6. click on train model and wait for the model to be trained

About the Project:

This project aims to create an automated system for classifying music genres using Convolutional Neural Networks (CNNs). The system processes audio files by extracting MFCC (Mel-Frequency Cepstral Coefficients) features and trains a CNN model to accurately detect genres. The GUI enables users to train the model and input audio files for genre classification. The goal is to streamline music organization and improve recommendation systems by efficiently categorizing audio files by genre.