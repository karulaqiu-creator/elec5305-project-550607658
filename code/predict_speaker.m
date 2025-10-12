clc; clear; close all;

%% Load trained model
dataFolder = 'C:\Users\karula qiu\Desktop\ELEC5305-Project\car';
modelPath = fullfile(dataFolder, 'optimized_speaker_model.mat');
if ~isfile(modelPath), error('Trained model file not found: %s', modelPath); end
load(modelPath,'modelData');

net = modelData.net;
normParams = modelData.normParams;
numCoeffs = modelData.numCoeffs;
fs = modelData.fs;
frameSize = modelData.frameSize;
frameStep = modelData.frameStep;
maxFrames = modelData.maxFrames;
speakerCategories = modelData.categories;

%% Select audio file to predict
[fileName, filePath] = uigetfile('*.wav','Select an audio file for speaker prediction');
if isequal(fileName,0), error('No file selected'); end
fileFullPath = fullfile(filePath,fileName);

%% Read and preprocess audio
[audio, fs_read] = audioread(fileFullPath);
if fs_read ~= fs
    audio = resample(audio, fs, fs_read);
end

audio = preprocessAudio(audio, fs);

%% Extract features
mfcc = extractAdvancedMFCC(audio, fs, frameSize, frameStep, numCoeffs, maxFrames);

%% Convert to 4D tensor and normalize
sampleData = reshape(mfcc, [numCoeffs, maxFrames, 1, 1]);
sampleNorm = applyNormalization(sampleData, normParams);

%% Convert to sequence for LSTM
sampleSeq = {squeeze(sampleNorm(:,:,1,1))};

%% Predict speaker
predLabel = classify(net, sampleSeq);
predProb = max(softmax(net.predict(sampleSeq{1})));

fprintf('Predicted speaker: %s (Confidence: %.2f%%)\n', string(predLabel), predProb*100);

%% Plot MFCC and prediction
figure('Position',[100,100,800,400]);
subplot(2,1,1);
imagesc(mfcc); axis xy;
xlabel('Frame'); ylabel('MFCC Coefficient');
title(sprintf('MFCC Features - Predicted Speaker: %s', string(predLabel)));
colorbar;

subplot(2,1,2);
bar(softmax(net.predict(sampleSeq{1})));
xticks(1:numel(speakerCategories));
xticklabels(speakerCategories);
ylabel('Probability'); xlabel('Speaker');
title('Prediction Probabilities');
ylim([0 1]);
grid on;

%% --- Functions ---
% Use exactly the same functions as in training/extract_features
function audio = preprocessAudio(audio, fs)
    audio = audio - mean(audio);
    preEmphasis = 0.97;
    audio = filter([1 -preEmphasis],1,audio);
    [b,a] = butter(6,[300 3400]/(fs/2),'bandpass');
    audio = filter(b,a,audio);
    if max(abs(audio))>0, audio=audio/max(abs(audio))*0.95; end
    frameLength=512; hopLength=256;
    numFrames=floor((length(audio)-frameLength)/hopLength)+1;
    energy=zeros(1,numFrames); zcr=zeros(1,numFrames);
    for i=1:numFrames
        idx=(i-1)*hopLength+1:(i-1)*hopLength+frameLength;
        frame=audio(idx); energy(i)=sum(frame.^2);
        zcr(i)=sum(abs(diff(sign(frame))))/(2*frameLength);
    end
    energyThresh=median(energy)+0.5*std(energy);
    zcrThresh=median(zcr)+0.5*std(zcr);
    speechFrames=(energy>energyThresh)|(zcr>zcrThresh);
    speechFrames=medfilt1(double(speechFrames),5)>0;
    if any(speechFrames)
        startFrame=find(speechFrames,1,'first');
        endFrame=find(speechFrames,1,'last');
        startSample=max((startFrame-1)*hopLength+1,1);
        endSample=min(endFrame*hopLength,length(audio));
        audio=audio(startSample:endSample);
    end
end

function mfccs = extractAdvancedMFCC(audio, fs, frameSize, frameStep, numCoeffs, maxFrames)
    baseCoeffs=floor(numCoeffs/3);
    basicMfcc=extractBasicMFCC(audio,fs,frameSize,frameStep,baseCoeffs,maxFrames);
    deltaMfcc=computeDelta(basicMfcc);
    deltaDeltaMfcc=computeDelta(deltaMfcc);
    mfccs=[basicMfcc;deltaMfcc;deltaDeltaMfcc];
    if size(mfccs,1)~=numCoeffs
        mfccs=mfccs(1:min(numCoeffs,size(mfccs,1)),:);
        if size(mfccs,1)<numCoeffs
            mfccs=[mfccs;zeros(numCoeffs-size(mfccs,1),size(mfccs,2))];
        end
    end
    if size(mfccs,2)<maxFrames
        mfccs=[mfccs,zeros(size(mfccs,1),maxFrames-size(mfccs,2))];
    else
        mfccs=mfccs(:,1:maxFrames);
    end
end

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
    frames=frames.*hamming(frameLen);
    NFFT=2^nextpow2(frameLen);
    magFFT=abs(fft(frames,NFFT));
    numFilters=26; melFilter=createMelFilterBank(fs,NFFT,numFilters);
    filterOut=melFilter*magFFT(1:NFFT/2+1,:);
    filterOut=max(filterOut,eps);
    mfccs=dct(log(filterOut));
    mfccs=mfccs(1:min(numCoeffs,size(mfccs,1)),:);
    if size(mfccs,2)<maxFrames
        mfccs=[mfccs,zeros(size(mfccs,1),maxFrames-size(mfccs,2))];
    else
        mfccs=mfccs(:,1:maxFrames);
    end
end

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

function normalizedData=applyNormalization(data,params)
    [numCoeffs,numFrames,numChannels,numSamples]=size(data);
    normalizedData=data;
    for i=1:numSamples
        sample=reshape(data(:,:,:,i),[],1);
        normalizedSample=(sample-params.mean)./max(params.std,eps);
        normalizedData(:,:,:,i)=reshape(normalizedSample,numCoeffs,numFrames,numChannels);
    end
end