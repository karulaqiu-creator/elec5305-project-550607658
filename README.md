# 🎙️ELEC5305 Project – Speaker Recognition & Keyword Spotting

This repository contains the implementation of my ELEC5305 project, which has two main parts:

1. **Speaker recognition** using a CNN + BiLSTM model in MATLAB.  
2. **Keyword spotting (KWS)** using a lightweight CNN–Transformer model in Python.

🗣️ 1. Speaker Recognition (MATLAB)

### 1.1. Prerequisites

- MATLAB (R2021b or later is recommended)
- 🔧Toolboxes:
  - Signal Processing Toolbox
  - Deep Learning Toolbox
  - Audio Toolbox

📥1.2. Step 1 – Download the speaker dataset
Script: download_ears_dataset.m

This script downloads and prepares the speaker recognition dataset.

How to run:

Open MATLAB.
Set the current folder to the project root.
Run:download_ears_dataset

The script will:
Download the EARS (or similar) speaker dataset.
Unpack it into the project’s data directory (see comments in the script for the exact folder).
Arrange files into subfolders per speaker (each speaker has their own folder of .wav files).

🎼1.3. Step 2 – Extract MFCC features
Script: extract_features.m
This script computes MFCC-based features for all audio files and saves them into a single .mat file used for training.

Run extract_features

After this step you should see a file like:
allFeatures.mat
saved in your dataset folder (path is defined inside extract_features.m).
It typically contains:
allFeatures – 4D tensor of MFCC features
allLabels – speaker labels
fs, frameSize, frameStep, maxFrames, etc.

🤖1.4. Step 3 – Train the speaker recognition model
Script: balanced_final_train.m
This script Loads allFeatures.mat builds a CNN + BiLSTM architecture,applies data augmentation and global normalization,trains the classifier with a class-balanced loss,saves the trained model as speaker_model_v5_balanced_final_fixed.mat

Run balanced_final_train

During training you will see:
Training progress window (loss, accuracy curves)
Final train and test accuracy printed in the MATLAB console and confusion matrix
At the end, a file like:
speaker_model_v5_balanced_final_fixed.mat
is saved into your data or project folder, containing:
net – trained network
normParam – normalization parameters
Feature configuration (e.g., numCoeffs, fs, frameSize, …)
Train/test accuracy

🔍1.5. Step 4 – Run the speaker prediction demo
Example script: examples/predict_speaker.m

This script Loads speaker_model_v5_balanced_final_fixed.mat
Lets you choose a .wav file via a file dialog
Applies the same preprocessing and MFCC extraction pipeline
Classifies the speaker and visualises the result

Run predict_speaker

What will happen:
A file selection dialog pops up – choose a .wav file containing speech.
The script:
Resamples the audio if needed
Applies preprocessing (DC removal, pre-emphasis, bandpass, VAD)
Extracts MFCC + Δ + ΔΔ features
Normalises them with the stored normParam
Passes them to the trained CNN–BiLSTM model
In the MATLAB command window you will see:
text
Predicted speaker: <speaker_id> (Confidence: XX.XX%)
A figure window shows:
The MFCC feature map for the selected utterance
A bar plot with the predicted probability for each speaker

🔑2. Keyword Spotting (KWS) – Python
The second part of this project implements a keyword spotting system using a lightweight CNN–Transformer model trained on a speech commands dataset.

✔️2.1. Prerequisites
Python 3.8 or later
PyTorch (CPU or GPU)
Other dependencies (see requirements.txt or import statements in the code), for example:
numpy
scipy
souNddevice or pyaudio (for real-time audio)
librosa or torchaudio (for feature extraction)

🏋️‍♂️2.2. Step 1 – Train the KWS model
Script: kws_training.py
This script:
Loads the keyword dataset (e.g., a speech commands dataset)
Builds a CNN–Transformer keyword spotting network
Trains the model
Saves learned weights to kws_cnn_transformer.pth

Run kws_training.py

Typical behaviour:
Training loss and accuracy printed to the terminal
Optionally, validation accuracy / confusion matrix
At the end of training a file like:
kws_cnn_transformer.pth
is created in the output or models/ directory (check the script for the exact path).

🎤2.3. Step 2 – Real-time keyword detection
Script: Keyword_Detection.py
This script:
Loads the trained model from kws_cnn_transformer.pth
Opens the microphone / audio input stream
Continuously listens to audio
Applies the same feature extraction used during training
Runs the CNN–Transformer model on sliding windows
Prints or visualises detected keywords and timestamps
Before running:
Make sure the path to kws_cnn_transformer.pth inside Keyword_Detection.py is correct.
Check / configure:
device index for your microphone (if the script supports it)
the list of target keywords

Run python Keyword_Detection.py

What you should see:
The script will start capturing audio from your microphone.
When you say one of the trained keywords, the program will:
Display the detected keyword in the console, and/or
Show probability scores or a simple UI (depending on how you implemented it).
Stop the script with enter in the terminal.

3. 🧾Summary
Speaker recognition (MATLAB):
download_ears_dataset.m – download dataset
extract_features.m – extract MFCC features
balanced_final_train.m – train CNN + BiLSTM speaker model and save speaker_model_v5_balanced_final_fixed.mat
examples/predict_speaker.m – demo: select a .wav file and predict the speaker
Keyword spotting (Python):
kws_training.py – train CNN–Transformer KWS model and save kws_cnn_transformer.pth
Keyword_Detection.py – load the model and run real-time keyword detection from the microphone
If you run into issues (paths, dependencies, or dataset setup), please check the comments in each script and adjust directory paths accordingly.

