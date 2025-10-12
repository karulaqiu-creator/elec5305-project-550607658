clc; clear; close all;

%% Load extracted features
dataFolder = 'C:\Users\karula qiu\Desktop\ELEC5305-Project\car';
featureFile = 'C:\Users\karula qiu\Desktop\ELEC5305-Project\car\allFeatures.mat';
if ~isfile(featureFile), error('Feature file not found: %s', featureFile); end
load(featureFile, 'allFeatures','allLabels','numCoeffs','fs','frameSize','frameStep','maxFrames');

%% Convert labels to categorical type
allLabels = categorical(allLabels);
numClasses = numel(categories(allLabels));

%% Train-test split (85% training, 15% test)
cv = cvpartition(allLabels,'HoldOut',0.15);
trainIdx = cv.training; 
testIdx = cv.test;

trainData = allFeatures(:,:,:,trainIdx); % Training feature tensors
testData  = allFeatures(:,:,:,testIdx);  % Test feature tensors
trainLabels = allLabels(trainIdx);       % Training labels
testLabels  = allLabels(testIdx);        % Test labels

%% Normalize features
[trainDataNorm, normParams] = normalizeFeatures(trainData); % Compute mean/std and normalize train set
testDataNorm = applyNormalization(testData, normParams);    % Apply same normalization to test set

fprintf('Train size: %d, Test size: %d\n', sum(trainIdx), sum(testIdx));

%% Convert 4D tensors to cell array sequences for LSTM
trainSeq = arrayfun(@(i) squeeze(trainDataNorm(:,:,1,i)), 1:size(trainDataNorm,4), 'UniformOutput', false);
testSeq  = arrayfun(@(i) squeeze(testDataNorm(:,:,1,i)), 1:size(testDataNorm,4), 'UniformOutput', false);

%% Define CNN + BiLSTM network architecture
layers = [
    sequenceInputLayer(numCoeffs,'Name','input','MinLength',maxFrames) % Input: MFCC features
    convolution1dLayer(3,64,'Padding','same','Name','conv1')             % 1D Conv layer with 64 filters
    batchNormalizationLayer('Name','bn1')                                % Batch normalization
    reluLayer('Name','relu1')                                            % ReLU activation
    convolution1dLayer(3,64,'Padding','same','Name','conv2')             % Second 1D Conv layer
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    bilstmLayer(128,'OutputMode','last','Name','bilstm1')                % Bidirectional LSTM with 128 units
    dropoutLayer(0.5)                                                    % Dropout for regularization
    fullyConnectedLayer(numClasses,'Name','fc_final')                    % Fully connected layer
    softmaxLayer('Name','softmax')                                       % Softmax for classification probabilities
    classificationLayer('Name','class')];                                 % Classification output

%% Training options
options = trainingOptions('adam', ...
    'MaxEpochs',100, ...                          % Train for 100 epochs
    'MiniBatchSize',32, ...                       % Mini-batch size
    'InitialLearnRate',1e-3, ...                  % Initial learning rate
    'Shuffle','every-epoch', ...                  % Shuffle data each epoch
    'ValidationData',{testSeq, testLabels}, ...   % Use test set for validation
    'ValidationFrequency',30, ...
    'Plots','training-progress', ...
    'Verbose',true);

%% Train network
tic;
net = trainNetwork(trainSeq, trainLabels, layers, options);

%% Evaluate performance
trainPred = classify(net, trainSeq); % Predictions on training set
testPred  = classify(net, testSeq);  % Predictions on test set
trainAcc = mean(trainPred == trainLabels); % Training accuracy
testAcc  = mean(testPred == testLabels);  % Test accuracy
fprintf('Train Accuracy: %.2f%%\n', trainAcc*100);
fprintf('Test Accuracy: %.2f%%\n', testAcc*100);

%% Confusion matrix visualization for test set
figure;
confusionchart(testLabels, testPred);
title('Test Set Confusion Matrix');

%% Save model and relevant statistics
modelData = struct();
modelData.net = net;                  % Trained network
modelData.normParams = normParams;    % Normalization parameters
modelData.numCoeffs = numCoeffs;
modelData.fs = fs;
modelData.frameSize = frameSize;
modelData.frameStep = frameStep;
modelData.maxFrames = maxFrames;
modelData.categories = categories(trainLabels);
modelData.trainAccuracy = trainAcc;
modelData.testAccuracy  = testAcc;

modelPath = fullfile(dataFolder, 'speaker_model.mat');
save(modelPath, 'modelData', '-v7.3');
fprintf('Model saved to: %s\n', modelPath);

%% Summary and visualization
figure('Position',[100,100,1200,800]);

% Test confusion matrix
subplot(2,2,1);
confusionchart(testLabels,testPred,'Title','Test Confusion Matrix');

% Per-speaker accuracy bar chart
subplot(2,2,2);
classAccuracy = zeros(1, numClasses);
classNames = categories(testLabels);

for i = 1:numClasses
    thisClass = classNames{i};
    idx = testLabels == thisClass;                   % Logical index for this class
    classAccuracy(i) = mean(testPred(idx) == thisClass); % Compute per-class accuracy
end
bar(classAccuracy*100);
title('Per-Speaker Accuracy'); xlabel('Speaker'); ylabel('Accuracy (%)'); ylim([0 100]); grid on;

% Text summary of training
trainingTime = toc;
subplot(2,2,[3,4]);
text(0.1,0.7,'Model Training Summary','FontSize',14,'FontWeight','bold');
text(0.1,0.6,sprintf('• Dataset size: %d samples', sum(trainIdx)+sum(testIdx)),'FontSize',12);
text(0.1,0.5,sprintf('• Num speakers: %d', numClasses),'FontSize',12);
text(0.1,0.4,sprintf('• Train accuracy: %.2f%%', trainAcc*100),'FontSize',12);
text(0.1,0.3,sprintf('• Test accuracy: %.2f%%', testAcc*100),'FontSize',12);
text(0.1,0.2,sprintf('• Training time: %.1f min', trainingTime/60),'FontSize',12);
text(0.1,0.1,'• Network: 2 conv layers + BiLSTM','FontSize',12);
axis off;

%% Assign results to MATLAB workspace
assignin('base','trainedModel',modelData);
assignin('base','testAccuracy',testAcc);
assignin('base','trainAccuracy',trainAcc);

fprintf('Training complete. Model saved to workspace.\n');

%% --- Functions ---
% Normalize features by computing mean and std per coefficient
function [normalizedData, params] = normalizeFeatures(data)
    [numCoeffs,numFrames,numChannels,numSamples]=size(data);
    allData=reshape(data,[],numSamples);
    params.mean=mean(allData,2); 
    params.std=std(allData,0,2);
    params.median=median(allData,2); 
    params.mad=mad(allData,1,2);
    normalizedData = data;
    for i=1:numSamples
        sample=reshape(data(:,:,:,i),[],1);
        normalizedSample=(sample-params.mean)./max(params.std,eps);
        normalizedData(:,:,:,i)=reshape(normalizedSample,numCoeffs,numFrames,numChannels);
    end
end

% Apply existing normalization parameters to new data
function normalizedData = applyNormalization(data, params)
    [numCoeffs,numFrames,numChannels,numSamples]=size(data);
    normalizedData=data;
    for i=1:numSamples
        sample=reshape(data(:,:,:,i),[],1);
        normalizedSample=(sample-params.mean)./max(params.std,eps);
        normalizedData(:,:,:,i)=reshape(normalizedSample,numCoeffs,numFrames,numChannels);
    end
end