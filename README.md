# Elec5305-project
Project Overview:
This project implements an optimized speaker recognition system using deep learning techniques. The system extracts advanced MFCC features, including Δ and ΔΔ coefficients, from speech recordings and employs a hybrid CNN-LSTM architecture for classification. Data augmentation is applied to increase robustness, leveraging time-stretching, pitch-shifting, and volume control. The goal is to achieve speaker identification accuracy above 95%.\n

Current Progress / Achievements:   
Dataset preparation:   
This project uses EARS (Expressive Anechoic Recordings of Speech) as dataset to do the model training and evaluation of speaker recognition and keyword detection. This dataset high-quality speech recordings from multiple speakers, making it suitable for deep learning speech recognition tasks.
The process of the dataset preparation is as follows:
1.	Downloading dataset: Write a MATLAB script to download the compressed files (such as p001.zip, p002.zip, etc.) for each speaker directly from the official GitHub repository and automatically decompress them to a local directory. 
2.	Subset Selection: To conserve storage space and training time, only 15 speakers are kept and each speakers have approximately 160 to ensure balanced data across speakers.
3.	Blind Test Set: Separately, download a separate "blind test set" to evaluate model performance on unseen data.  
4.	Data Organization: The data is organized by speaker, with each speaker corresponding to a subfolder:
   <img width="573" height="178" alt="image" src="https://github.com/user-attachments/assets/f4f3b628-1244-4fee-b8f7-019b7522dcb9" />

Pre-processing: Speech recordings organized by speaker with preprocessing (DC removal, pre-emphasis, normalization).  
Feature extraction: Basic and advanced MFCC extraction implemented with handling of variable frame lengths.  
Data augmentation: Integrated audioDataAugmenter (if Audio Toolbox is available) for time-stretching, pitch-shifting, and volume control.  
Model architecture: Designed hybrid CNN-LSTM network for temporal and local feature learning.  
Training pipeline: Implemented training with data normalization, stratified train/test split, and validation monitoring.  
Evaluation: Confusion matrix and per-class accuracy visualization included.  

Model saving: Trained network and normalization parameters saved for future use.   
