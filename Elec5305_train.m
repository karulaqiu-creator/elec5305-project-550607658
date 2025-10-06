

clc; clear; close all;
restoredefaultpath;  
rehash toolboxcache; 
%% ==============================
%% 1. Set random seed for reproducibility
%% ==============================
try
    rng(2024, 'twister'); % Use fixed seed
catch
    rng(2024); % Fallback for older MATLAB versions
end

%% ==============================
%% 2. Dataset configuration
%% ==============================
dataFolder = 'C:\Users\karula qiu\Desktop\ELEC5305-Project\car';

if ~isfolder(dataFolder)
    error('Dataset folder not found: %s\nPlease ensure the folder exists.', dataFolder);
end

% List speaker folders
speakers = dir(dataFolder);
speakers = speakers([speakers.isdir] & ~ismember({speakers.name}, {'.','..'}));

if isempty(speakers)
    error('No speaker folders found in %s', dataFolder);
end

fprintf('Found %d speaker folders.\n', length(speakers));

%% ==============================
%% 3. Feature & preprocessing configuration
%% ==============================
fs = 16000;           % Target sampling rate
frameSize = 0.032;    % Frame size in seconds (32ms)
frameStep = 0.016;    % Frame hop (16ms)
numCoeffs = 39;       % MFCC coefficients (13 + Δ + ΔΔ)
maxFrames = 150;      % Maximum number of frames per sample

%% ==============================
%% 4. Data augmentation setup
%% ==============================
% Use Audio Toolbox if available
augmenter = [];
tbList = ver;
tbNames = {tbList.Name};
if any(strcmpi(tbNames,'Audio Toolbox'))
    try
        % augmenter = audioDataAugmenter( ...
        %     'AddNoise', true, ...
        %     'SNRRange', [10 30], ...
        %     'TimeStretch', [0.85 1.15], ...
        %     'PitchShift', [-3 3], ...
        %     'VolumeControl', [0.7 1.3]);
        % disp('Audio Toolbox detected. Advanced augmentation enabled.');
        augmenter = audioDataAugmenter( ...
            'AugmentationMode','independent', ...        % or 'sequential'
            'TimeStretchProbability',0.5, ...
            'PitchShiftProbability',0.5, ...
            'VolumeControlProbability',0.5);
        disp('Audio Toolbox detected. Advanced augmentation enabled.');
    catch ME
        warning('Failed to create audioDataAugmenter: %s\nUsing basic augmentation.', ME.message);
        augmenter = [];
    end
else
    warning('Audio Toolbox not installed. Using basic augmentation fallback.');
end

%% ==============================
%% 5. Feature extraction & augmentation
%% ==============================
allFeatures = [];
allLabels = {};
fileCount = 0;

fprintf('Starting feature extraction...\n');
tic;

for spkIdx = 1:numel(speakers)
    speaker = speakers(spkIdx).name;
    files = dir(fullfile(dataFolder, speaker, '*.wav'));
    
    if isempty(files)
        warning('No WAV files for speaker %s', speaker);
        continue;
    end
    
    fprintf('Processing speaker %s (%d files)...\n', speaker, length(files));
    
    for fIdx = 1:numel(files)
        filePath = fullfile(files(fIdx).folder, files(fIdx).name);
        %try
            [audio, fs_read] = audioread(filePath);
            
            % Resample if needed
            if fs_read ~= fs
                audio = resample(audio, fs, fs_read);
            end
            
            % Preprocess audio (DC removal, normalization)
            audio = preprocessAudio(audio);
            
            % Extract MFCC features
            mfccFeat = extractAdvancedMFCC(audio, fs, frameSize, frameStep, numCoeffs, maxFrames);
            
            % Store original
            allFeatures = cat(4, allFeatures, reshape(mfccFeat, [numCoeffs, maxFrames, 1, 1]));
            allLabels = [allLabels; speaker];
            fileCount = fileCount + 1;
            
            % Augmentation: 2 variants per original
            if ~isempty(augmenter)
                for aIdx = 1:2
                    audioAug = augment(augmenter, audio, fs);
                    mfccAug = extractAdvancedMFCC(audioAug, fs, frameSize, frameStep, numCoeffs, maxFrames);
                    allFeatures = cat(4, allFeatures, reshape(mfccAug, [numCoeffs, maxFrames, 1, 1]));
                    allLabels = [allLabels; speaker];
                    fileCount = fileCount + 1;
                end
            end
            
        % catch ME
        %     warning('Failed to process %s: %s', filePath, ME.message);
        % end
    end
end

fprintf('Feature extraction complete: %d samples. Elapsed time: %.1f s\n', fileCount, toc);

%% ==============================
%% 6. Data preparation
%% ==============================
allLabels = categorical(allLabels);
numClasses = numel(categories(allLabels));

if fileCount < 100
    error('Insufficient data (%d samples). Need at least 100.', fileCount);
end

% Stratified split (85% train, 15% test)
cv = cvpartition(allLabels, 'HoldOut', 0.15);
trainData = allFeatures(:,:,:,cv.training);
trainLabels = allLabels(cv.training);
testData = allFeatures(:,:,:,cv.test);
testLabels = allLabels(cv.test);

% Feature normalization
[trainDataNorm, normParams] = normalizeFeatures(trainData);
testDataNorm = applyNormalization(testData, normParams);

fprintf('Train size: %d, Test size: %d\n', sum(cv.training), sum(cv.test));

%% ==============================
%% 7. Define CNN architecture (ResNet-inspired)
%% ==============================
numHiddenUnits = 128; % LSTM hidden units

layers = [
    % Input: MFCC features as sequences [numCoeffs x maxFrames]
    sequenceInputLayer(numCoeffs, 'Name', 'input')
    
    % 1D convolution along time axis for local feature extraction
    convolution1dLayer(3,64,'Padding','same','Name','conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    
    convolution1dLayer(3,64,'Padding','same','Name','conv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    maxPooling1dLayer(2,'Stride',2,'Name','pool1')
    
    % LSTM layer to capture temporal dependencies
    lstmLayer(numHiddenUnits,'OutputMode','last','Name','lstm1')
    
    % Fully connected layers for classification
    fullyConnectedLayer(256,'Name','fc1')
    reluLayer('Name','relu_fc1')
    dropoutLayer(0.3,'Name','dropout1')
    
    fullyConnectedLayer(numClasses,'Name','fc_final')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','class')
];

%% ==============================
%% 8. Training options
%% ==============================
options = trainingOptions('adam', ...
    'MaxEpochs',150, ...
    'MiniBatchSize',32, ...        % Smaller batch for LSTM memory
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',30, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{testDataNorm,testLabels}, ...
    'ValidationFrequency',30, ...
    'ValidationPatience',20, ...
    'Plots','training-progress', ...
    'Verbose',true, ...
    'ExecutionEnvironment','auto');

%% ==============================
%% 9. Train the network
%% ==============================
fprintf('Starting hybrid CNN-LSTM training...\n');
trainingStartTime = tic;
try
    net = trainNetwork(trainDataNorm, trainLabels, layers, options);
    trainingTime = toc(trainingStartTime);
    fprintf('Training complete. Elapsed time: %.1f minutes\n', trainingTime/60);
catch ME
    error('Training failed: %s\nCheck data dimensions and toolbox installation.', ME.message);
end

%% ==============================
%% 10. Evaluate performance
%% ==============================
trainPred = classify(net, trainDataNorm);
testPred = classify(net, testDataNorm);

trainAcc = mean(trainPred == trainLabels);
testAcc = mean(testPred == testLabels);

fprintf('Train Accuracy: %.2f%%\n', trainAcc*100);
fprintf('Test Accuracy: %.2f%%\n', testAcc*100);

%% ==============================
%% 11. Save model & normalization parameters
%% ==============================
modelData.net = net;
modelData.normParams = normParams;
modelData.categories = categories(trainLabels);
modelData.trainAccuracy = trainAcc;
modelData.testAccuracy = testAcc;

save(fullfile(dataFolder,'optimized_speaker_model.mat'), 'modelData', '-v7.3');

%% ==============================
%% 12. Plot evaluation results
%% ==============================
figure('Position',[100 100 1200 800]);

% Confusion matrix
subplot(2,2,1);
confusionchart(testLabels,testPred,'Title','Test Confusion Matrix');

% Per-class accuracy
subplot(2,2,2);
classNames = categories(testLabels);
classAcc = zeros(numClasses,1);
for i=1:numClasses
    idx = testLabels == classNames{i};
    if sum(idx) > 0
        classAcc(i) = mean(testPred(idx) == testLabels(idx));
    end
end
bar(classAcc*100);
title('Per-Speaker Accuracy');
xlabel('Speaker ID'); ylabel('Accuracy (%)');
ylim([80 100]); grid on;

sgtitle('Optimized CNN Speaker Recognition Performance','FontSize',16);

%% ==============================
%% 13. Export results to workspace
%% ==============================
assignin('base','trainedModel',modelData);
assignin('base','trainAccuracy',trainAcc);
assignin('base','testAccuracy',testAcc);

fprintf('✅ Training completed! Model saved and workspace variables updated.\n');

%Pre-processing
function audio = preprocessAudio(audio, fs)
    % AUDIO PREPROCESSING
    % 1. Remove DC
    audio = audio - mean(audio);

    % 2. Pre-emphasis
    preEmphasis = 0.97;
    audio = filter([1 -preEmphasis], 1, audio);

    % 3. Optional bandpass filtering (300-3400 Hz)
    if nargin > 1 && ~isempty(fs)
        [b, a] = butter(6, [300 3400]/(fs/2), 'bandpass');
        audio = filter(b, a, audio);
    end

    % 4. Normalize to [-0.95, 0.95]
    if max(abs(audio)) > 0
        audio = audio / max(abs(audio)) * 0.95;
    end

    % 5. Frame parameters
    frameLength = 512;
    hopLength = 256;

    % 6. Compute short-time energy (STE) and zero-crossing rate (ZCR)
    numFrames = floor((length(audio)-frameLength)/hopLength)+1;
    energy = zeros(1,numFrames);
    zcr = zeros(1,numFrames);

    for i = 1:numFrames
        idx = (i-1)*hopLength + 1 : (i-1)*hopLength + frameLength;
        frame = audio(idx);
        energy(i) = sum(frame.^2);
        zcr(i) = sum(abs(diff(sign(frame)))) / (2*frameLength);
    end

    % 7. Voice activity detection (VAD) using adaptive threshold
    energyThresh = median(energy) + 0.5*std(energy);
    zcrThresh = median(zcr) + 0.5*std(zcr);

    speechFrames = (energy > energyThresh) | (zcr > zcrThresh);

    % 8. Median filter to smooth detection
    speechFrames = medfilt1(double(speechFrames), 5) > 0;

    % 9. Convert frame indices to sample indices
    if any(speechFrames)
        startFrame = find(speechFrames, 1, 'first');
        endFrame = find(speechFrames, 1, 'last');
        startSample = max((startFrame-1)*hopLength+1, 1);
        endSample = min(endFrame*hopLength, length(audio));
        audio = audio(startSample:endSample);
    end
end


function mfccs = extractAdvancedMFCC(audio, fs, frameSize, frameStep, numCoeffs, maxFrames)
% Extract advanced MFCC features including delta and delta-delta
% Input:
%   audio      - audio vector
%   fs         - sampling rate
%   frameSize  - frame size in seconds
%   frameStep  - frame step in seconds
%   numCoeffs  - total number of coefficients (MFCC + Δ + ΔΔ)
%   maxFrames  - maximum number of frames
% Output:
%   mfccs      - MFCC feature matrix [numCoeffs x maxFrames]

    % 1. Basic MFCC
    baseCoeffs = floor(numCoeffs / 3); % assume Δ and ΔΔ included
    basicMfcc = extractBasicMFCC(audio, fs, frameSize, frameStep, baseCoeffs, maxFrames);
    
    % 2. Delta
    deltaMfcc = computeDelta(basicMfcc);
    
    % 3. Delta-Delta
    deltaDeltaMfcc = computeDelta(deltaMfcc);
    
    % 4. Concatenate
    mfccs = [basicMfcc; deltaMfcc; deltaDeltaMfcc];
    
    % 5. Adjust rows if needed
    if size(mfccs,1) ~= numCoeffs
        mfccs = mfccs(1:min(numCoeffs, size(mfccs,1)), :);
        if size(mfccs,1) < numCoeffs
            mfccs = [mfccs; zeros(numCoeffs - size(mfccs,1), size(mfccs,2))];
        end
    end
    
    % 6. Adjust columns to maxFrames
    if size(mfccs,2) < maxFrames
        mfccs = [mfccs, zeros(size(mfccs,1), maxFrames - size(mfccs,2))];
    else
        mfccs = mfccs(:, 1:maxFrames);
    end
end

function mfccs = extractBasicMFCC(audio, fs, frameSize, frameStep, numCoeffs, maxFrames)
% Extract basic MFCC features from an audio signal
% Input:
%   audio      - audio vector
%   fs         - sampling rate
%   frameSize  - frame size in seconds
%   frameStep  - frame step in seconds
%   numCoeffs  - number of MFCC coefficients
%   maxFrames  - maximum number of frames
% Output:
%   mfccs      - MFCC feature matrix [numCoeffs x maxFrames]

    % Convert table to array if needed
    if istable(audio)
        audio = table2array(audio);
    end
    
    % Frame parameters
    frameLen = round(frameSize * fs);
    stepLen = round(frameStep * fs);
    numFrames = max(floor((length(audio) - frameLen) / stepLen) + 1, 1);
    
    % Preallocate frame matrix
    frames = zeros(frameLen, numFrames);
    
    % Split audio into frames
    for i = 1:numFrames
        startIdx = (i-1) * stepLen + 1;
        endIdx = min(startIdx + frameLen - 1, length(audio));
        frames(1:(endIdx-startIdx+1), i) = audio(startIdx:endIdx);
    end
    
    % Apply Hamming window
    frames = frames .* hamming(frameLen);
    
    % FFT
    NFFT = 2^nextpow2(frameLen);
    magFFT = abs(fft(frames, NFFT));
    
    % Mel filterbank
    numFilters = 26;
    melFilter = createMelFilterBank(fs, NFFT, numFilters);
    
    % Apply filterbank
    filterOut = melFilter * magFFT(1:NFFT/2+1, :);
    filterOut = max(filterOut, eps); % Avoid log(0)
    
    % DCT to get MFCC
    mfccs = dct(log(filterOut));
    
    % Keep only desired coefficients
    mfccs = mfccs(1:min(numCoeffs, size(mfccs,1)), :);
    
    % Pad or truncate to maxFrames
    if size(mfccs,2) < maxFrames
        mfccs = [mfccs, zeros(size(mfccs,1), maxFrames - size(mfccs,2))];
    else
        mfccs = mfccs(:, 1:maxFrames);
    end
end

function delta = computeDelta(features)
    [numCoeffs, numFrames] = size(features);
    delta = zeros(numCoeffs, numFrames);
    
    for t = 1:numFrames
        if t == 1
            delta(:, t) = features(:, 2) - features(:, 1);
        elseif t == numFrames
            delta(:, t) = features(:, numFrames) - features(:, numFrames-1);
        else
            delta(:, t) = (features(:, t+1) - features(:, t-1)) / 2;
        end
    end
end

function melFilter = createMelFilterBank(fs, NFFT, numFilters)
    
    lowFreq = 80;
    highFreq = fs / 2;
    
    lowMel = 2595 * log10(1 + lowFreq / 700);
    highMel = 2595 * log10(1 + highFreq / 700);
    
    melPoints = linspace(lowMel, highMel, numFilters + 2);
    hzPoints = 700 * (10.^(melPoints / 2595) - 1);
    
    bin = floor((NFFT + 1) * hzPoints / fs);
    
    melFilter = zeros(numFilters, NFFT/2 + 1);
    
    for m = 1:numFilters
        left = bin(m);
        center = bin(m + 1);
        right = bin(m + 2);
        
        for k = left:center
            if center > left
                melFilter(m, k + 1) = (k - left) / (center - left);
            end
        end
        
        for k = center:right
            if right > center
                melFilter(m, k + 1) = (right - k) / (right - center);
            end
        end
    end
end

function [normalizedData, params] = normalizeFeatures(data)
    
    [numCoeffs, numFrames, numChannels, numSamples] = size(data);
    
    allData = reshape(data, [], numSamples);
    params.mean = mean(allData, 2);
    params.std = std(allData, 0, 2);
    params.median = median(allData, 2);
    params.mad = mad(allData, 1, 2);
    
    normalizedData = data;
    for i = 1:numSamples
        sample = reshape(data(:, :, :, i), [], 1);
        normalizedSample = (sample - params.mean) ./ max(params.std, eps);
        normalizedData(:, :, :, i) = reshape(normalizedSample, numCoeffs, numFrames, numChannels);
    end
end

function normalizedData = applyNormalization(data, params)
    
    [numCoeffs, numFrames, numChannels, numSamples] = size(data);
    normalizedData = data;
    
    for i = 1:numSamples
        sample = reshape(data(:, :, :, i), [], 1);
        normalizedSample = (sample - params.mean) ./ max(params.std, eps);
        normalizedData(:, :, :, i) = reshape(normalizedSample, numCoeffs, numFrames, numChannels);
    end
end 