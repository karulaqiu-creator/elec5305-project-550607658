%% =====================================================================
% ELEC5305 - Speaker Recognition
% Verification Script (Fully Corrected Version)
% Author: Kyle Qiu
% =====================================================================
clc; clear; close all;

%% 1) Load trained model
dataFolder = 'C:\Users\karula qiu\OneDrive - The University of Sydney (Students)\桌面\ELEC5305-Project\car';
modelPath = fullfile(dataFolder, 'speaker_model_v5_balanced_final_fixed.mat');

if ~isfile(modelPath)
    error('❌ Trained model not found: %s', modelPath);
end

load(modelPath, 'modelData');

% Extract required components
net          = modelData.net;
normParam    = modelData.normParam;      % ✔ correct field name
numCoeffs    = modelData.numCoeffs;
fs           = modelData.fs;
frameSize    = modelData.frameSize;
frameStep    = modelData.frameStep;
maxFrames    = modelData.maxFrames;
speakerCategories = modelData.categories;

fprintf("✅ Loaded trained model successfully.\n");

%% 2) Select audio file
[fileName, filePath] = uigetfile('*.wav','Select an audio file for prediction');
if isequal(fileName,0)
    error('❌ No audio file selected.');
end
fileFullPath = fullfile(filePath, fileName);

fprintf("🎧 Selected file: %s\n", fileFullPath);

%% 3) Read and preprocess audio
[audio, fs_read] = audioread(fileFullPath);

% Resample if needed
if fs_read ~= fs
    fprintf("🔄 Resampling from %d → %d Hz...\n", fs_read, fs);
    audio = resample(audio, fs, fs_read);
end

% Preprocess audio
audio = preprocessAudio(audio, fs);

% Ensure minimum duration
if length(audio) < round(0.2 * fs)
    audio = [audio; zeros(round(0.2*fs)-length(audio), 1)];
end

%% 4) Extract MFCC features
mfcc = extractAdvancedMFCC(audio, fs, frameSize, frameStep, numCoeffs, maxFrames);

%% 5) Convert to 4D + normalization
sampleData = reshape(mfcc, [numCoeffs, maxFrames, 1, 1]);

% ✔ Use same normalization used during training
sampleNorm = applyGlobal(sampleData, normParam);

%% 6) Convert to LSTM sequence
sampleSeq = { squeeze(sampleNorm(:,:,1,1)) };

%% 7) Predict
[YPred, scores] = classify(net, sampleSeq);

predLabel = string(YPred);
predProb  = max(scores);

fprintf("\n================ Prediction Result ================\n");
fprintf("👤 Predicted Speaker: %s\n", predLabel);
fprintf("📊 Confidence: %.2f%%\n", predProb * 100);
fprintf("===================================================\n");

%% 8) Plot MFCC + probability distribution
figure('Position',[100,100,900,450]);

subplot(2,1,1);
imagesc(mfcc); axis xy; colormap jet;
title(sprintf('MFCC Features (Prediction: %s)', predLabel));
xlabel('Frame'); ylabel('MFCC Coefficient'); colorbar;

subplot(2,1,2);
bar(scores, 'FaceColor',[0.2 0.5 0.8]);
xticks(1:numel(speakerCategories));
xticklabels(speakerCategories);
xtickangle(45);
ylabel('Probability'); ylim([0 1]); grid on;
title('Speaker Prediction Probabilities');

%% =====================================================================
% Supporting Functions (Same Style as Training)
% =====================================================================

function audio = preprocessAudio(audio, fs)
    % DC removal
    audio = audio - mean(audio);

    % Pre-emphasis
    preEmphasis = 0.97;
    audio = filter([1 -preEmphasis], 1, audio);

    % Bandpass filter
    [b,a] = butter(6, [300 3400] / (fs/2), 'bandpass');
    audio = filter(b,a,audio);

    % Peak normalize
    if max(abs(audio)) > 0
        audio = audio / max(abs(audio)) * 0.95;
    end

    % VAD
    frameLength = 512; hopLength = 256;
    numFrames = floor((length(audio)-frameLength)/hopLength)+1;
    energy = zeros(1,numFrames); zcr = zeros(1,numFrames);

    for i = 1:numFrames
        idx = (i-1)*hopLength+1 : (i-1)*hopLength+frameLength;
        frame = audio(idx);
        energy(i) = sum(frame.^2);
        zcr(i) = sum(abs(diff(sign(frame))))/(2*frameLength);
    end

    energyThresh = median(energy) + 0.5*std(energy);
    zcrThresh = median(zcr) + 0.5*std(zcr);
    speechFrames = (energy > energyThresh) | (zcr > zcrThresh);
    speechFrames = medfilt1(double(speechFrames),5)>0;

    if any(speechFrames)
        startFrame = find(speechFrames,1,'first');
        endFrame   = find(speechFrames,1,'last');
        audio = audio((startFrame-1)*hopLength+1 : endFrame*hopLength);
    end
end


function dataNorm = applyGlobal(data, param)
    sz = size(data);
    A = reshape(data, [], 1);
    Ahat = (A - param.mean) ./ param.std;
    dataNorm = reshape(Ahat, sz);
end

function mfccs = extractAdvancedMFCC(audio, fs, frameSize, frameStep, numCoeffs, maxFrames)
    baseCoeffs=floor(numCoeffs/3);
    basicMfcc=extractBasicMFCC(audio,fs,frameSize,frameStep,baseCoeffs,maxFrames); % Base MFCC
    deltaMfcc=computeDelta(basicMfcc);        % First derivative
    deltaDeltaMfcc=computeDelta(deltaMfcc);   % Second derivative
    mfccs=[basicMfcc;deltaMfcc;deltaDeltaMfcc]; 
    % Pad or truncate to match numCoeffs
    if size(mfccs,1)~=numCoeffs
        mfccs=mfccs(1:min(numCoeffs,size(mfccs,1)),:);
        if size(mfccs,1)<numCoeffs
            mfccs=[mfccs;zeros(numCoeffs-size(mfccs,1),size(mfccs,2))];
        end
    end
    % Pad or truncate to maxFrames
    if size(mfccs,2)<maxFrames
        mfccs=[mfccs,zeros(size(mfccs,1),maxFrames-size(mfccs,2))];
    else
        mfccs=mfccs(:,1:maxFrames);
    end
end

% Extract basic MFCC from frames
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
    magFFT=abs(fft(frames,NFFT));
    numFilters=26; melFilter=createMelFilterBank(fs,NFFT,numFilters); 
    filterOut=melFilter*magFFT(1:NFFT/2+1,:); 
    filterOut=max(filterOut,eps);
    mfccs=dct(log(filterOut)); 
    mfccs=mfccs(1:min(numCoeffs,size(mfccs,1)),:);
    % Pad or truncate frames
    if size(mfccs,2)<maxFrames
        mfccs=[mfccs,zeros(size(mfccs,1),maxFrames-size(mfccs,2))];
    else
        mfccs=mfccs(:,1:maxFrames);
    end
end

% Compute delta (first derivative)
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

% Create Mel filterbank
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
