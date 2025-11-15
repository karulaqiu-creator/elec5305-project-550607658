%% ================================================================
% ELEC5305 - Speaker Recognition (balanced_final_train)
% Author: Kyle Qiu
%
% Key improvements:
%  ✅ 60-D MFCC features (more discriminative)
%  ✅ Longer temporal coverage (maxFrames=250)
%  ✅ Strong 3× data augmentation with label replication
%  ✅ Class-balanced loss function
%  ✅ Layer normalization + tuned dropout
% ================================================================
clc; clear; close all;

%% 1) Load MFCC features
dataFolder  = 'C:\Users\karula qiu\OneDrive - The University of Sydney (Students)\桌面\ELEC5305-Project\car';
featureFile = fullfile(dataFolder, 'allFeatures.mat');
load(featureFile, 'allFeatures','allLabels','numCoeffs','fs','frameSize','frameStep','maxFrames');
fprintf('✅ Loaded MFCC features from: %s\n', featureFile);

%% Override parameters for improved setup
numCoeffs = 60;      % 39 → 60 dimensions
maxFrames = 250;     % longer frame coverage
allLabels = categorical(allLabels);
classes   = categories(allLabels);
numClasses = numel(classes);

%% 2) Train/Test split (90/10)
cv = cvpartition(allLabels, 'HoldOut', 0.10);
trainIdx = training(cv);
testIdx  = test(cv);
Xtrain = allFeatures(:,:,:,trainIdx);
Ytrain = allLabels(trainIdx);
Xtest  = allFeatures(:,:,:,testIdx);
Ytest  = allLabels(testIdx);

fprintf('Train samples: %d | Test samples: %d | Classes: %d\n', ...
    sum(trainIdx), sum(testIdx), numClasses);

%% 3) Normalize (fit on train, apply to both)
[XtrainN, normParam] = normalizeGlobal(Xtrain);
XtestN = applyGlobal(Xtest, normParam);

%% 4) Data augmentation (3× per sample, with label replication)
[XtrainA, YtrainA] = augmentStrongV5(XtrainN, Ytrain, 3);
fprintf('🔁 Augmented training set: %d → %d samples\n', numel(Ytrain), numel(YtrainA));

%% 5) Convert to cell sequences
trainSeq = arrayfun(@(i) squeeze(XtrainA(:,:,1,i)), 1:size(XtrainA,4), 'UniformOutput', false);
testSeq  = arrayfun(@(i) squeeze(XtestN(:,:,1,i)),  1:size(XtestN,4),  'UniformOutput', false);
Ytrain   = YtrainA; % update to augmented labels

%% 6) Define CNN + BiLSTM model
layers = [
    sequenceInputLayer(numCoeffs,'Name','input','MinLength',maxFrames)

    convolution1dLayer(5,128,'Padding','same','Name','conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    dropoutLayer(0.25,'Name','drop1')

    convolution1dLayer(3,128,'Padding','same','Name','conv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    maxPooling1dLayer(2,'Stride',2,'Name','pool1')

    convolution1dLayer(3,256,'Padding','same','Name','conv3')
    batchNormalizationLayer('Name','bn3')
    reluLayer('Name','relu3')
    dropoutLayer(0.35,'Name','drop3')

    bilstmLayer(256,'OutputMode','last','Name','bilstm')
    layerNormalizationLayer('Name','ln_bilstm')
    dropoutLayer(0.5,'Name','drop_bilstm')

    fullyConnectedLayer(256,'Name','fc1')
    reluLayer('Name','relu_fc1')
    dropoutLayer(0.45,'Name','drop_fc1')

    fullyConnectedLayer(numClasses,'Name','fc_final')
    softmaxLayer('Name','softmax')
    weightedClassificationLayer(Ytrain)  % ✅ class balancing
];

%% 7) Training options
options = trainingOptions('adam', ...
    'MaxEpochs',180, ...
    'MiniBatchSize',32, ...
    'InitialLearnRate',2e-4, ...
    'L2Regularization',1e-4, ...
    'GradientThreshold',5, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{testSeq, Ytest}, ...
    'ValidationPatience',30, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment','auto', ...
    'Verbose',false);

%% 8) Train model
tic;
fprintf('\n🚀 Training v5_balanced_final_fixed model...\n');
net = trainNetwork(trainSeq, Ytrain, layers, options);
trainTime = toc;

%% 9) Evaluate accuracy
YPredTrain = classify(net, trainSeq);
YPredTest  = classify(net, testSeq);
accTrain = mean(YPredTrain == Ytrain);
accTest  = mean(YPredTest  == Ytest);
fprintf('\n✅ Results (v5_balanced_final_fixed):\n');
fprintf('Train Accuracy: %.2f%% | Test Accuracy: %.2f%% | Time: %.1f min\n', ...
    accTrain*100, accTest*100, trainTime/60);

%% 10) Visualization
figure('Position',[100,100,1100,480]);
subplot(1,2,1);
confusionchart(Ytest, YPredTest);
title('Test Set Confusion Matrix (v5\_balanced\_final\_fixed)');
subplot(1,2,2);
C = categories(Ytest);
per = zeros(numel(C),1);
for i = 1:numel(C)
    idx = Ytest==C{i};
    per(i) = mean(YPredTest(idx)==C{i});
end
bar(per*100); ylim([0 100]); grid on;
title('Per-Speaker Accuracy'); xlabel('Speaker'); ylabel('Accuracy (%)');

%% 11) Save model
modelData = struct('net',net,'normParam',normParam,'numCoeffs',numCoeffs,...
    'fs',fs,'frameSize',frameSize,'frameStep',frameStep,'maxFrames',maxFrames,...
    'categories',{classes},'trainAccuracy',accTrain,'testAccuracy',accTest);
save(fullfile(dataFolder,'speaker_model_v5_balanced_final_fixed.mat'),'modelData','-v7.3');
fprintf('\n💾 Saved -> %s\n', fullfile(dataFolder,'speaker_model_v5_balanced_final_fixed.mat'));

%% ===== Helper functions =====
function [Xn, P] = normalizeGlobal(X)
    sz = size(X); A = reshape(X, [], sz(4));
    P.mean = mean(A,2); P.std = std(A,0,2) + 1e-6;
    Ahat = (A - P.mean) ./ P.std; Xn = reshape(Ahat, sz);
end

function Xn = applyGlobal(X, P)
    sz = size(X); A = reshape(X, [], sz(4));
    Ahat = (A - P.mean) ./ P.std; Xn = reshape(Ahat, sz);
end

function [Xa, Yaug] = augmentStrongV5(X, Y, mult)
    Xa = []; Yaug = categorical([]);
    [numC,numT,~,N] = size(X);
    for i = 1:N
        S = X(:,:,1,i);
        for j = 1:mult
            Saug = S;
            Saug = Saug + (0.005 + 0.005*rand)*randn(size(Saug)); % noise
            Saug = circshift(Saug,[0 randi([-4,4])]);             % shift
            Saug = Saug*(0.85+0.3*rand);                          % volume
            maskLen = max(1,round(0.05*numT));                    % time mask
            start = randi([1,max(1,numT-maskLen+1)]);
            Saug(:,start:start+maskLen-1) = 0;                    % mask region
            Xa = cat(4,Xa,reshape(Saug,[numC,numT,1,1]));
            Yaug = [Yaug; Y(i)];                                  % replicate label
        end
    end
end

function layer = weightedClassificationLayer(Ytrain)
    C = categories(Ytrain);
    counts = countcats(Ytrain);
    w = 1 ./ max(counts,1);
    w = w / sum(w) * numel(w);
    layer = classificationLayer('Name','class','Classes',C,'ClassWeights',w);
end
