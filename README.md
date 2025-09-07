# elec5305-project-550607658
1.	Project Title
Voiceprint recognition and keyword spotting based on deep learning
2.	Project Overview
This project is focused on realizing a speech processing system,that can recognize speaker identity and detect keywords in speech.By recording a small set of audio samples,extracting Mel-spectrogram features,and use CNN to do the Voiceprint recognition and keyword recognition.This project will perform audio features extraction,deep learning model training and real-time speech recognition.
Current speech interaction technologies have been widely applied in smart homes,voice assistants and security systems,but the following problems still exist:
1.Lack of personalization:most existing speech recognition systems focus only on speech content and cannot distinguish between different speakers.
2.Noise interference:recognition accuracy drops significantly in real-world noisy environments.
3.Real-time constraints:traditional large-scale models are computationally expensive and unsuitable for lightweight, real-time applications.
The significance of this project lies in combining knowledge from ELEC5305(speech signal processing) with deep learning for real-world applications.The deliverable will be a real-time system which may contribute to personalized speech interaction, intelligent control,and multi-user recognition systems.
3.	Background and Motivation
With the development of AI and audio processing technique,voiceprint recognition has become increasingly popular in applications such as smart devices, speech assistants, and security verification.While most existing solutions tend to depend on large datasets or computational resources,but for small scaled and personalized speech recognition,there are still some challenges to be resolved.
This project aims to combine voiceprint recognition and keyword recognition to develop a real-time speech processing system which can distinguish the speaker’s identity and detecting specific voice commands.The project concentrates on the following applications and values:
1.Personalized speech recognition:distinguishing between different users through voiceprint recognition to enhance system security and interaction experience.
2.Keyword detection and human-computer interaction:by means of detecting specific commands to enable rapid response to user instructions and supports smart device control.
3.Robustness in noisy environments:by combining denoising, filtering, short-time features and deep learning,the system aims to achieve stable performance in real-world conditions.
4.Integration of course knowledge:this project applies concepts covered in ELEC5305,like mel-frequency cepstral coefficients feature extraction and short-time fourier transform to a complete practical engineering workflow.
Through this project,I can consolidate theoretical knowledge learned from course and gain hands-on experience in data collection,signal pre-processing,feature extraction,deep learning model training and evaluation,laying a foundation for extending the system to multi-user real-time recognition or more advanced keyword spotting tasks.
4.	Proposed Methodology
Tools and platforms:MATLAB
Signal processing techniques:1.MFCC(Mel-Frequency Cepstral Coefficients)-Extract related frequency features from speech,as the input of the recognition model.
2.STFT(Short-Time Fourier Transform)–Performs time-frequency analysis and visualizes how the frequency content of the speech signals changes over time.
3.CNN(Convolutional Neural Network)–A deep learning model that takes MFCC features as input to perform classification tasks,for speaker recognition (distinguishing my voice from others) and keyword detection.
Data sources:Google Speech Commands Dataset and a self recorded dataset.
5.	Expected Outcomes
Use MATLAB to accomplish the whole process of speech sampling,pre-processing,feature-extraction and deep learning classifications.
Performance metrics:
Speaker recognition accuracy:the probability of correctly distinguishing my speech from others. 
Keyword detection accuracy:the recognition rate of specific commands (e.g.‘keyword’) and response latency.
GitHub documentation and demo:
Provide the integral code,signal processing and model training instructions,including the visualizing results.
