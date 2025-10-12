# Elec5305-project
## 🎙️ Speaker Recognition Project
Project Overview:
This project implements an optimized speaker recognition system using deep learning techniques. The system extracts advanced MFCC features, including Δ and ΔΔ coefficients, from speech recordings and employs a hybrid CNN-LSTM architecture for classification. Data augmentation is applied to increase robustness, leveraging time-stretching, pitch-shifting, and volume control. The goal is to achieve speaker identification accuracy above 95%.

Current Progress / Achievements:   
Dataset preparation:   
This project uses EARS (Expressive Anechoic Recordings of Speech) as dataset to do the model training and evaluation of speaker recognition and keyword detection. This dataset high-quality speech recordings from multiple speakers, making it suitable for deep learning speech recognition tasks.
The process of the dataset preparation is as follows:
1.	Downloading dataset: Write a MATLAB script to download the compressed files (such as p001.zip, p002.zip, etc.) for each speaker directly from the official GitHub repository and automatically decompress them to a local directory. 
2.	Subset Selection: To conserve storage space and training time, only 15 speakers are kept and each speakers have approximately 160 to ensure balanced data across speakers.
3.	Data Organization: The data is organized by speaker, with each speaker corresponding to a subfolder:

dataset/

├── speaker1

├── speaker2

├── speaker3

.

.

.

└── speaker15

Pre-processing: Speech recordings organized by speaker with preprocessing (DC removal, pre-emphasis, normalization).  
<img width="250" height="200" alt="1" src="https://github.com/user-attachments/assets/afb463e4-9207-44ed-9521-7ad70098a69e" />

Feature extraction: Basic and advanced MFCC extraction implemented with handling of variable frame lengths.  
<img width="250" height="200" alt="image" src="https://github.com/user-attachments/assets/f96166c2-d3f0-426c-a940-f0486727b399" />

Data augmentation: Integrated audioDataAugmenter (if Audio Toolbox is available) for time-stretching, pitch-shifting, and volume control.  
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/8b80fa8d-7327-4883-94b1-ae9acce2cc3c" />


Model architecture: Designed hybrid CNN-LSTM network for temporal and local feature learning.  
<img width="250" height="200" alt="image" src="https://github.com/user-attachments/assets/89d00718-975f-4912-85ca-545c08900680" />

Training pipeline: Implemented training with data normalization, stratified train/test split, and validation monitoring.  
<img width="190" height="300" alt="image" src="https://github.com/user-attachments/assets/fd71402a-da74-42df-bc9e-6850dd543a25" />

Evaluation: Confusion matrix and per-class accuracy visualization included.

<img width="190" height="300" alt="image" src="https://github.com/user-attachments/assets/b4f2cb1f-23fe-48f0-8293-44182e90f52f" />

Model saving: Trained network and normalization parameters saved for future use.    

### 📁 Folder Structure
- `code/`: Training, feature extraction, and prediction scripts.
- `models/`: Trained model (`.mat`) file.
- `samples/`: Example input and output samples for demonstration.


### 🧩 Example
This example shows how to predict the speaker of a given audio file using the trained model.

1. **Input audio file:**  
`samples/input_sample.wav`

2. **Run the prediction in MATLAB:**  
```matlab
`Load and predict speaker
predict_speaker('samples/input_sample.wav');


%References:  
96syh. "matlab-speaker-recognition" GitHub, 2025, https://github.com/96syh/matlab-speaker-recognition. Accessed 6 Oct. 2025.  
Ko, T., Peddinti, V., Povey, D., & Khudanpur, S., 2015. Audio augmentation for speech recognition. Interspeech 2015, pp. 3586–3589.  
Snyder, D., Garcia-Romero, D., Sell, G., Povey, D., & Khudanpur, S., 2018. X-vectors: Robust DNN embeddings for speaker recognition. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 5329–5333.  
Tiwari, M., & Verma, D. K. (2024). Enhanced text-independent speaker recognition using MFCC, Bi-LSTM, and CNN-based noise removal techniques. International Journal of Speech Technology, 27(4), 1013–1026.  

