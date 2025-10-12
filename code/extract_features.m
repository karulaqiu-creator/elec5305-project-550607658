clc; clear; close all; 

%% Set random seed for reproducibility
try
    rng(2024, 'twister');  % Modern way to set seed
catch
    rng(2024);              % Fallback for older MATLAB versions
end

%% Data folder configuration
dataFolder = 'C:\Users\karula qiu\Desktop\ELEC5305-Project\car';
if ~isfolder(dataFolder)
    error('Dataset folder does not exist: %s', dataFolder);
end

% List all speaker directories (exclude '.' and '..')
speakers = dir(dataFolder);
speakers = speakers([speakers.isdir] & ~ismember({speakers.name},{'.','..'}));
fprintf('Found %d speaker directories\n', length(speakers));

%% Parameters for audio processing
fs = 16000;           % Target sampling frequency
frameSize = 0.032;    % Frame length in seconds
frameStep = 0.016;    % Frame step in seconds (hop size)
numCoeffs = 39;       % Number of MFCC coefficients (including delta & delta-delta)
maxFrames = 150;      % Maximum number of frames per audio sample

%% Optional data augmentation
try
    augmenter = audioDataAugmenter( ...
        'AddNoise', true, ...          % Add background noise
        'SNRRange', [10, 30], ...     % Noise signal-to-noise ratio range
        'TimeStretch', [0.85, 1.15],... % Speed up / slow down audio
        'PitchShift', [-3, 3], ...    % Shift pitch up/down in semitones
        'VolumeControl', [0.7, 1.3]); % Random volume change
    disp('Advanced data augmentation enabled');
catch
    warning('Audio Toolbox not available, skipping augmentation');
    augmenter = [];
end

%% Feature extraction loop
allFeatures = [];  % Store all MFCC feature tensors
allLabels = {};    % Store corresponding speaker labels
fileCount = 0;     % Counter for total processed files

disp('Extracting features...');
tic;
for spkIdx = 1:numel(speakers)
    speaker = speakers(spkIdx).name;
    files = dir(fullfile(dataFolder, speaker, '*.wav')); % Find all WAV files for this speaker
    
    if isempty(files)
        warning('No WAV files found for speaker %s', speaker);
        continue;
    end
    
    fprintf('Processing speaker %s (%d files)...\n', speaker, length(files));
    
    for fileIdx = 1:numel(files)
        filePath = fullfile(files(fileIdx).folder, files(fileIdx).name);
        try
            [audio, fs_read] = audioread(filePath); % Read audio
            if fs_read ~= fs
                audio = resample(audio, fs, fs_read); % Resample if needed
            end
            
            audio = preprocessAudio(audio, fs); % Preprocess audio (VAD, DC removal, etc.)
            mfcc = extractAdvancedMFCC(audio, fs, frameSize, frameStep, numCoeffs, maxFrames); % Extract MFCC + delta
            
            % Append feature and label
            allFeatures = cat(4, allFeatures, reshape(mfcc,[numCoeffs,maxFrames,1,1]));
            allLabels = [allLabels; speaker];
            fileCount = fileCount + 1;
            
            % Apply optional data augmentation
            if ~isempty(augmenter)
                for augIdx = 1:2  % Create 2 augmented versions
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
allLabels = categorical(allLabels);  % Convert string labels to categorical type
featureFile = fullfile(dataFolder,'allFeatures.mat');
save(featureFile, 'allFeatures', 'allLabels', 'numCoeffs', 'fs', 'frameSize', 'frameStep', 'maxFrames', '-v7.3');
fprintf('Features, labels and parameters saved to: %s\n', featureFile);

%% --- Functions (same as training) ---
% Preprocess audio: DC removal, pre-emphasis, bandpass, normalize, VAD
function audio = preprocessAudio(audio, fs)
    audio = audio - mean(audio);                % Remove DC
    preEmphasis = 0.97;
    audio = filter([1 -preEmphasis],1,audio);  % Apply pre-emphasis
    [b,a] = butter(6,[300 3400]/(fs/2),'bandpass'); % Bandpass 300-3400 Hz
    audio = filter(b,a,audio);
    if max(abs(audio))>0, audio=audio/max(abs(audio))*0.95; end % Normalize amplitude
    
    % Voice Activity Detection (VAD)
    frameLength=512; hopLength=256;
    numFrames=floor((length(audio)-frameLength)/hopLength)+1;
    energy=zeros(1,numFrames); zcr=zeros(1,numFrames);
    for i=1:numFrames
        idx=(i-1)*hopLength+1:(i-1)*hopLength+frameLength;
        frame=audio(idx); 
        energy(i)=sum(frame.^2);                        % Short-time energy
        zcr(i)=sum(abs(diff(sign(frame))))/(2*frameLength); % Zero-crossing rate
    end
    energyThresh=median(energy)+0.5*std(energy);       % Energy threshold
    zcrThresh=median(zcr)+0.5*std(zcr);               % ZCR threshold
    speechFrames=(energy>energyThresh)|(zcr>zcrThresh);
    speechFrames=medfilt1(double(speechFrames),5)>0;  % Smooth VAD
    
    if any(speechFrames)
        startFrame=find(speechFrames,1,'first');
        endFrame=find(speechFrames,1,'last');
        startSample=max((startFrame-1)*hopLength+1,1);
        endSample=min(endFrame*hopLength,length(audio));
        audio=audio(startSample:endSample); % Keep speech-only segment
    end
end

% Extract MFCC + delta + delta-delta
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