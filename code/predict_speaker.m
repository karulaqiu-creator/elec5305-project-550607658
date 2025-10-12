clc; clear; close all;

%% Load trained model
dataFolder = 'C:\Users\karula qiu\Desktop\ELEC5305-Project\car';
modelPath = fullfile(dataFolder, 'speaker_model.mat');
if ~isfile(modelPath), error('Trained model file not found: %s', modelPath); end
load(modelPath,'modelData');  % Load trained model data

% Extract model variables
net = modelData.net;              % Trained CNN+BiLSTM network
normParams = modelData.normParams; % Normalization parameters
numCoeffs = modelData.numCoeffs;   % Number of MFCC coefficients
fs = modelData.fs;                 % Sampling frequency
frameSize = modelData.frameSize;   % Frame size in seconds
frameStep = modelData.frameStep;   % Frame step in seconds
maxFrames = modelData.maxFrames;   % Maximum number of frames
speakerCategories = modelData.categories; % List of speaker labels

%% Select audio file to predict
[fileName, filePath] = uigetfile('*.wav','Select an audio file for speaker prediction');
if isequal(fileName,0), error('No file selected'); end
fileFullPath = fullfile(filePath,fileName);

%% Read and preprocess audio
[audio, fs_read] = audioread(fileFullPath); % Read WAV file
if fs_read ~= fs
    audio = resample(audio, fs, fs_read);  % Resample if necessary
end

audio = preprocessAudio(audio, fs);        % Preprocess audio (DC removal, pre-emphasis, VAD, etc.)

%% Extract features
mfcc = extractAdvancedMFCC(audio, fs, frameSize, frameStep, numCoeffs, maxFrames); % Compute MFCC + delta + delta-delta

%% Convert to 4D tensor and normalize
sampleData = reshape(mfcc, [numCoeffs, maxFrames, 1, 1]); % Convert to 4D tensor for network
sampleNorm = applyNormalization(sampleData, normParams);  % Apply previously computed normalization

%% Convert to sequence for LSTM
sampleSeq = {squeeze(sampleNorm(:,:,1,1))}; % Convert 4D tensor to sequence cell array for LSTM input

%% Predict speaker
predLabel = classify(net, sampleSeq); % Classify the speaker
predProb = max(softmax(net.predict(sampleSeq{1}))); % Compute confidence

fprintf('Predicted speaker: %s (Confidence: %.2f%%)\n', string(predLabel), predProb*100);

%% Plot MFCC and prediction
figure('Position',[100,100,800,400]);

subplot(2,1,1);
imagesc(mfcc); axis xy;  % Plot MFCC coefficients
xlabel('Frame'); ylabel('MFCC Coefficient');
title(sprintf('MFCC Features - Predicted Speaker: %s', string(predLabel)));
colorbar;

subplot(2,1,2);
bar(softmax(net.predict(sampleSeq{1}))); % Plot probability distribution
xticks(1:numel(speakerCategories));
xticklabels(speakerCategories);
ylabel('Probability'); xlabel('Speaker');
title('Prediction Probabilities');
ylim([0 1]);
grid on;

%% --- Functions ---
% All functions are copied from training/extract_features scripts
% 1. Preprocess audio: DC removal, pre-emphasis, bandpass, normalization, VAD
function audio = preprocessAudio(audio, fs)
    audio = audio - mean(audio);          % Remove DC
    preEmphasis = 0.97;
    audio = filter([1 -preEmphasis],1,audio); % Apply pre-emphasis
    [b,a] = butter(6,[300 3400]/(fs/2),'bandpass'); % Bandpass filter
    audio = filter(b,a,audio);
    if max(abs(audio))>0, audio=audio/max(abs(audio))*0.95; end % Normalize amplitude
    frameLength=512; hopLength=256;
    numFrames=floor((length(audio)-frameLength)/hopLength)+1;
    energy=zeros(1,numFrames); zcr=zeros(1,numFrames);
    for i=1:numFrames
        idx=(i-1)*hopLength+1:(i-1)*hopLength+frameLength;
        frame=audio(idx); 
        energy(i)=sum(frame.^2);                % Compute short-time energy
        zcr(i)=sum(abs(diff(sign(frame))))/(2*frameLength); % Compute zero-crossing rate
    end
    energyThresh=median(energy)+0.5*std(energy); % Adaptive energy threshold
    zcrThresh=median(zcr)+0.5*std(zcr);         % Adaptive ZCR threshold
    speechFrames=(energy>energyThresh)|(zcr>zcrThresh); % VAD
    speechFrames=medfilt1(double(speechFrames),5)>0;   % Smooth VAD using median filter
    if any(speechFrames)
        startFrame=find(speechFrames,1,'first');
        endFrame=find(speechFrames,1,'last');
        startSample=max((startFrame-1)*hopLength+1,1);
        endSample=min(endFrame*hopLength,length(audio));
        audio=audio(startSample:endSample); % Keep only detected speech
    end
end

% 2. Extract advanced MFCC (MFCC + delta + delta-delta)
function mfccs = extractAdvancedMFCC(audio, fs, frameSize, frameStep, numCoeffs, maxFrames)
    baseCoeffs=floor(numCoeffs/3);
    basicMfcc=extractBasicMFCC(audio,fs,frameSize,frameStep,baseCoeffs,maxFrames); % Base MFCC
    deltaMfcc=computeDelta(basicMfcc);     % First-order difference
    deltaDeltaMfcc=computeDelta(deltaMfcc); % Second-order difference
    mfccs=[basicMfcc;deltaMfcc;deltaDeltaMfcc];
    if size(mfccs,1)~=numCoeffs
        mfccs=mfccs(1:min(numCoeffs,size(mfccs,1)),:);
        if size(mfccs,1)<numCoeffs
            mfccs=[mfccs;zeros(numCoeffs-size(mfccs,1),size(mfccs,2))];
        end
    end
    if size(mfccs,2)<maxFrames
        mfccs=[mfccs,zeros(size(mfccs,1),maxFrames-size(mfccs,2))]; % Pad frames
    else
        mfccs=mfccs(:,1:maxFrames); % Truncate frames
    end
end

% 3. Extract basic MFCC from audio frames
function mfccs=extractBasicMFCC(audio,fs,frameSize,frameStep,numCoeffs,maxFrames)
    if istable(audio), audio=table2array(audio); end
    frameLen=round(frameSize*fs); stepLen=round(frameStep*fs);
    numFrames=max(floor((length(audio)-frameLen)/stepLen)+1,1);
    frames=zeros(frameLen,numFrames);
    for i=1:numFrames
        startIdx=(i-1)*stepLen+1;
        endIdx=min(startIdx+frameLen-1,length(audio));
        frames(1:(endIdx-startIdx+1),i)=audio(startIdx:endIdx);
    end
    frames=frames.*hamming(frameLen); % Apply Hamming window
    NFFT=2^nextpow2(frameLen);
    magFFT=abs(fft(frames,NFFT));     % Compute magnitude FFT
    numFilters=26; 
    melFilter=createMelFilterBank(fs,NFFT,numFilters); % Mel filterbank
    filterOut=melFilter*magFFT(1:NFFT/2+1,:); % Apply Mel filter
    filterOut=max(filterOut,eps);
    mfccs=dct(log(filterOut)); % DCT to get MFCC
    mfccs=mfccs(1:min(numCoeffs,size(mfccs,1)),:);
    if size(mfccs,2)<maxFrames
        mfccs=[mfccs,zeros(size(mfccs,1),maxFrames-size(mfccs,2))]; % Pad frames
    else
        mfccs=mfccs(:,1:maxFrames);
    end
end

% 4. Compute delta (first or second-order difference)
function delta=computeDelta(features)
    [numCoeffs,numFrames]=size(features);
    delta=zeros(numCoeffs,numFrames);
    for t=1:numFrames
        if t==1, delta(:,t)=features(:,2)-features(:,1);
        elseif t==numFrames, delta(:,t)=features(:,numFrames)-features(:,numFrames-1);
        else, delta(:,t)=(features(:,t+1)-features(:,t-1))/2;
        end
    end
end

% 5. Create Mel filter bank
function melFilter=createMelFilterBank(fs,NFFT,numFilters)
    lowFreq=80; highFreq=fs/2;
    lowMel=2595*log10(1+lowFreq/700);
    highMel=2595*log10(1+highFreq/700);
    melPoints=linspace(lowMel,highMel,numFilters+2);
    hzPoints=700*(10.^(melPoints/2595)-1);
    bin=floor((NFFT+1)*hzPoints/fs);
    melFilter=zeros(numFilters,NFFT/2+1);
    for m=1:numFilters
        left=bin(m); center=bin(m+1); right=bin(m+2);
        for k=left:center
            if center>left, melFilter(m,k+1)=(k-left)/(center-left); end
        end
        for k=center:right
            if right>center, melFilter(m,k+1)=(right-k)/(right-center); end
        end
    end
end

% 6. Apply normalization using precomputed mean/std
function normalizedData=applyNormalization(data,params)
    [numCoeffs,numFrames,numChannels,numSamples]=size(data);
    normalizedData=data;
    for i=1:numSamples
        sample=reshape(data(:,:,:,i),[],1);
        normalizedSample=(sample-params.mean)./max(params.std,eps); % Z-score normalization
        normalizedData(:,:,:,i)=reshape(normalizedSample,numCoeffs,numFrames,numChannels);
    end
end