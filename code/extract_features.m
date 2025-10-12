clc; clear; close all; 

%% Set random seed for reproducibility
try
    rng(2024, 'twister');  
catch
    rng(2024);  
end

%% Data folder configuration
dataFolder = 'C:\Users\karula qiu\Desktop\ELEC5305-Project\car';
if ~isfolder(dataFolder)
    error('Dataset folder does not exist: %s', dataFolder);
end

% List speaker directories
speakers = dir(dataFolder);
speakers = speakers([speakers.isdir] & ~ismember({speakers.name},{'.','..'}));
fprintf('Found %d speaker directories\n', length(speakers));

%% Parameters
fs = 16000;          
frameSize = 0.032;   
frameStep = 0.016;   
numCoeffs = 39;      
maxFrames = 150;     

%% Optional data augmentation
try
    augmenter = audioDataAugmenter( ...
        'AddNoise', true, ...
        'SNRRange', [10, 30], ...
        'TimeStretch', [0.85, 1.15], ...
        'PitchShift', [-3, 3], ...
        'VolumeControl', [0.7, 1.3]);
    disp('Advanced data augmentation enabled');
catch
    warning('Audio Toolbox not available, skipping augmentation');
    augmenter = [];
end

%% Feature extraction loop
allFeatures = [];
allLabels = {};
fileCount = 0;

disp('Extracting features...');
tic;
for spkIdx = 1:numel(speakers)
    speaker = speakers(spkIdx).name;
    files = dir(fullfile(dataFolder, speaker, '*.wav'));
    
    if isempty(files)
        warning('No WAV files found for speaker %s', speaker);
        continue;
    end
    
    fprintf('Processing speaker %s (%d files)...\n', speaker, length(files));
    
    for fileIdx = 1:numel(files)
        filePath = fullfile(files(fileIdx).folder, files(fileIdx).name);
        try
            [audio, fs_read] = audioread(filePath);
            if fs_read ~= fs
                audio = resample(audio, fs, fs_read);
            end
            
            audio = preprocessAudio(audio, fs); 
            mfcc = extractAdvancedMFCC(audio, fs, frameSize, frameStep, numCoeffs, maxFrames);
            
            allFeatures = cat(4, allFeatures, reshape(mfcc,[numCoeffs,maxFrames,1,1]));
            allLabels = [allLabels; speaker];
            fileCount = fileCount + 1;
            
            if ~isempty(augmenter)
                for augIdx = 1:2
                    audioAug = augment(augmenter,audio,fs);
                    mfccAug = extractAdvancedMFCC(audioAug, fs, frameSize, frameStep, numCoeffs, maxFrames);
                    allFeatures = cat(4, allFeatures, reshape(mfccAug,[numCoeffs,maxFrames,1,1]));
                    allLabels = [allLabels; speaker];
                    fileCount = fileCount + 1;
                end
            end
            
        catch ME
            warning('Failed to process file: %s (%s)', filePath, ME.message);
        end
    end
end
processingTime = toc;
fprintf('Feature extraction done: %d samples in %.1f seconds\n', fileCount, processingTime);

%% Convert labels to categorical and save features
allLabels = categorical(allLabels);
featureFile = fullfile(dataFolder,'allFeatures.mat');
save(featureFile, 'allFeatures', 'allLabels', 'numCoeffs', 'fs', 'frameSize', 'frameStep', 'maxFrames', '-v7.3');
fprintf('Features, labels and parameters saved to: %s\n', featureFile);

%% --- Functions (same as training) ---
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