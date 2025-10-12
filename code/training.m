clc; clear; close all;

%% Load extracted features
dataFolder = 'C:\Users\karula qiu\Desktop\ELEC5305-Project\car';
featureFile = 'C:\Users\karula qiu\Desktop\ELEC5305-Project\car\allFeatures.mat';
if ~isfile(featureFile), error('Feature file not found: %s', featureFile); end
load(featureFile, 'allFeatures','allLabels','numCoeffs','fs','frameSize','frameStep','maxFrames');

%% Convert labels to categorical
allLabels = categorical(allLabels);
numClasses = numel(categories(allLabels));

%% Train-test split (85%/15%)
cv = cvpartition(allLabels,'HoldOut',0.15);
trainIdx = cv.training; testIdx = cv.test;

trainData = allFeatures(:,:,:,trainIdx);
testData  = allFeatures(:,:,:,testIdx);
trainLabels = allLabels(trainIdx);
testLabels  = allLabels(testIdx);

%% Normalize features
[trainDataNorm, normParams] = normalizeFeatures(trainData);
testDataNorm = applyNormalization(testData, normParams);

fprintf('Train size: %d, Test size: %d\n', sum(trainIdx), sum(testIdx));

%% Convert 4D tensors to sequences for LSTM
trainSeq = arrayfun(@(i) squeeze(trainDataNorm(:,:,1,i)), 1:size(trainDataNorm,4), 'UniformOutput', false);
testSeq  = arrayfun(@(i) squeeze(testDataNorm(:,:,1,i)), 1:size(testDataNorm,4), 'UniformOutput', false);

%% Define CNN + BiLSTM network
layers = [
    sequenceInputLayer(numCoeffs,'Name','input','MinLength',maxFrames)
    convolution1dLayer(3,64,'Padding','same','Name','conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    convolution1dLayer(3,64,'Padding','same','Name','conv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    bilstmLayer(128,'OutputMode','last','Name','bilstm1')
    dropoutLayer(0.5)
    fullyConnectedLayer(numClasses,'Name','fc_final')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','class')];

%% Training options
options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'MiniBatchSize',32, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{testSeq, testLabels}, ...
    'ValidationFrequency',30, ...
    'Plots','training-progress', ...
    'Verbose',true);

%% Train network
tic;
net = trainNetwork(trainSeq, trainLabels, layers, options);

%% Evaluate performance
trainPred = classify(net, trainSeq);
testPred  = classify(net, testSeq);
trainAcc = mean(trainPred == trainLabels);
testAcc  = mean(testPred == testLabels);
fprintf('Train Accuracy: %.2f%%\n', trainAcc*100);
fprintf('Test Accuracy: %.2f%%\n', testAcc*100);

%% Confusion matrix
figure;
confusionchart(testLabels, testPred);
title('Test Set Confusion Matrix');

%% Save model and stats
modelData = struct();
modelData.net = net;
modelData.normParams = normParams;
modelData.numCoeffs = numCoeffs;
modelData.fs = fs;
modelData.frameSize = frameSize;
modelData.frameStep = frameStep;
modelData.maxFrames = maxFrames;
modelData.categories = categories(trainLabels);
modelData.trainAccuracy = trainAcc;
modelData.testAccuracy  = testAcc;

modelPath = fullfile(dataFolder, 'optimized_speaker_model.mat');
save(modelPath, 'modelData', '-v7.3');
fprintf('Model saved to: %s\n', modelPath);

%% Summary and visualization
figure('Position',[100,100,1200,800]);
subplot(2,2,1);
confusionchart(testLabels,testPred,'Title','Test Confusion Matrix');

subplot(2,2,2);
classAccuracy = zeros(1, numClasses);
classNames = categories(testLabels);

for i = 1:numClasses
    thisClass = classNames{i};
    idx = testLabels == thisClass;           % logical index of samples of this class
    classAccuracy(i) = mean(testPred(idx) == thisClass);  % compute per-class accuracy
end
bar(classAccuracy*100);
title('Per-Speaker Accuracy'); xlabel('Speaker'); ylabel('Accuracy (%)'); ylim([0 100]); grid on;

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

%% Assign results to workspace
assignin('base','trainedModel',modelData);
assignin('base','testAccuracy',testAcc);
assignin('base','trainAccuracy',trainAcc);

fprintf('Training complete. Model saved to workspace.\n');

%% --- Functions ---
% normalizeFeatures, applyNormalization 保留和 extract_features.m 中完全一致
function [normalizedData, params] = normalizeFeatures(data)
    [numCoeffs,numFrames,numChannels,numSamples]=size(data);
    allData=reshape(data,[],numSamples);
    params.mean=mean(allData,2); params.std=std(allData,0,2);
    params.median=median(allData,2); params.mad=mad(allData,1,2);
    normalizedData = data;
    for i=1:numSamples
        sample=reshape(data(:,:,:,i),[],1);
        normalizedSample=(sample-params.mean)./max(params.std,eps);
        normalizedData(:,:,:,i)=reshape(normalizedSample,numCoeffs,numFrames,numChannels);
    end
end

function normalizedData = applyNormalization(data, params)
    [numCoeffs,numFrames,numChannels,numSamples]=size(data);
    normalizedData=data;
    for i=1:numSamples
        sample=reshape(data(:,:,:,i),[],1);
        normalizedSample=(sample-params.mean)./max(params.std,eps);
        normalizedData(:,:,:,i)=reshape(normalizedSample,numCoeffs,numFrames,numChannels);
    end
end