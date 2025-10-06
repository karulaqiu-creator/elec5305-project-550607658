# Elec5305-project
Project Overview:
This project implements an optimized speaker recognition system using deep learning techniques. The system extracts advanced MFCC features, including Δ and ΔΔ coefficients, from speech recordings and employs a hybrid CNN-LSTM architecture for classification. Data augmentation is applied to increase robustness, leveraging time-stretching, pitch-shifting, and volume control. The goal is to achieve speaker identification accuracy above 95%.\n

Current Progress / Achievements:

Dataset preparation: Speech recordings organized by speaker with preprocessing (DC removal, pre-emphasis, normalization).\n  
Feature extraction: Basic and advanced MFCC extraction implemented with handling of variable frame lengths.
Data augmentation: Integrated audioDataAugmenter (if Audio Toolbox is available) for time-stretching, pitch-shifting, and volume control.
Model architecture: Designed hybrid CNN-LSTM network for temporal and local feature learning.
Training pipeline: Implemented training with data normalization, stratified train/test split, and validation monitoring.
Evaluation: Confusion matrix and per-class accuracy visualization included.
Model saving: Trained network and normalization parameters saved for future use.
