clc; clear; close all;

% ==============================================================
% STAGE 4 — Energy Ablation + Benchmark Comparison
% ==============================================================
% Uses the strict outer protocol established in Stage 3C.
%
% Models:
%   1) Temporal LSTM — all features
%   2) Temporal LSTM — Energy_Wh removed
%   3) Random Forest (bagged regression trees)
%   4) Boosted Trees (MATLAB LSBoost benchmark)
%   5) SVR
%
% IMPORTANT:
% - Outer TEST battery is untouched by scaling/training/stopping.
% - LSTM hyperparameters are read from Stage 3C selected configs.
% - LSTM uses past temporal windows to predict NEXT-cycle SOH.
% - Classical benchmarks receive the SAME flattened temporal windows,
%   so the comparison uses the same information horizon.
% - Capacity, SOH, Cycle and Battery are never predictors.
% - 10 random seeds are used for LSTM and tree ensembles.
% ==============================================================

clc;
load('NASA_corrected_features.mat','All');
cfg = readtable('stage3c_selected_configs.csv');

allFeatures = { ...
    'V_mean','V_std','V_range','V_slope', ...
    'T_mean','T_std','DeltaT','T_slope','Energy_Wh'};

noEnergyFeatures = { ...
    'V_mean','V_std','V_range','V_slope', ...
    'T_mean','T_std','DeltaT','T_slope'};

seeds = [42 123 2026 3407 9999 7 77 314 2718 8080];

maxEpochs = 200;
validationPatience = 15;

batteryNames = unique(All.Battery,'stable');

results = table();
predictionTable = table();
rr = 0;

fprintf('Starting Stage 4 benchmark + Energy ablation...\n\n');

for outer = 1:numel(batteryNames)

    testBattery = batteryNames(outer);
    devBatteries = batteryNames(batteryNames ~= testBattery);

    % Find Stage-3C hyperparameters selected WITHOUT this test cell.
    c = cfg(cfg.OuterTestBattery == testBattery,:);
    if isempty(c)
        error('No Stage-3C configuration found for %s.',testBattery);
    end
    c = c(1,:);

    w = c.Window;
    h = c.HiddenUnits;
    lr = c.LearningRate;
    bs = c.BatchSize;

    % Same deterministic early-stopping validation rule as Stage 3C.
    nCycles = zeros(numel(devBatteries),1);
    for k = 1:numel(devBatteries)
        nCycles(k) = sum(All.Battery == devBatteries(k));
    end
    [~,ord] = sort(nCycles,'ascend');
    valBattery = devBatteries(ord(ceil(numel(ord)/2)));
    trainBatteries = devBatteries(devBatteries ~= valBattery);

    TrainRaw = All(ismember(All.Battery,trainBatteries),:);
    ValRaw   = All(All.Battery == valBattery,:);
    TestRaw  = All(All.Battery == testBattery,:);

    fprintf('TEST %s | TRAIN %s,%s | VAL %s | window=%d\n', ...
        testBattery,trainBatteries(1),trainBatteries(2),valBattery,w);

    % ==========================================================
    % A) LSTM — ALL FEATURES
    % ==========================================================
    [TrainA,ValA,TestA] = scale_three(TrainRaw,ValRaw,TestRaw,allFeatures);
    [XTrainA,YTrainA,~] = make_temporal_windows(TrainA,allFeatures,w);
    [XValA,YValA,~] = make_temporal_windows(ValA,allFeatures,w);
    [XTestA,YTestA,testCycles] = make_temporal_windows(TestA,allFeatures,w);

    for si = 1:numel(seeds)
        seed = seeds(si);
        rng(seed,'twister');

        layers = make_lstm_layers(numel(allFeatures),h);
        valFrequency = max(1,floor(numel(YTrainA)/bs));

        options = trainingOptions('adam', ...
            'InitialLearnRate',lr, ...
            'MaxEpochs',maxEpochs, ...
            'MiniBatchSize',bs, ...
            'Shuffle','every-epoch', ...
            'ValidationData',{XValA,YValA}, ...
            'ValidationFrequency',valFrequency, ...
            'ValidationPatience',validationPatience, ...
            'Verbose',false,'Plots','none');

        net = trainNetwork(XTrainA,YTrainA,layers,options);
        pred = double(predict(net,XTestA,'MiniBatchSize',bs));
        pred = pred(:);

        [rmse,mae,r2,smape] = metrics(double(YTestA(:)),pred);

        rr = rr + 1;
        results = add_result(results,rr,testBattery,"LSTM_All",seed, ...
            w,h,lr,bs,rmse,mae,r2,smape);

        predictionTable = add_predictions(predictionTable,testBattery, ...
            "LSTM_All",seed,testCycles,YTestA,pred);
    end

    % ==========================================================
    % B) LSTM — WITHOUT ENERGY
    % Same hyperparameters: isolated feature ablation.
    % ==========================================================
    [TrainN,ValN,TestN] = scale_three(TrainRaw,ValRaw,TestRaw,noEnergyFeatures);
    [XTrainN,YTrainN,~] = make_temporal_windows(TrainN,noEnergyFeatures,w);
    [XValN,YValN,~] = make_temporal_windows(ValN,noEnergyFeatures,w);
    [XTestN,YTestN,testCyclesN] = make_temporal_windows(TestN,noEnergyFeatures,w);

    for si = 1:numel(seeds)
        seed = seeds(si);
        rng(seed,'twister');

        layers = make_lstm_layers(numel(noEnergyFeatures),h);
        valFrequency = max(1,floor(numel(YTrainN)/bs));

        options = trainingOptions('adam', ...
            'InitialLearnRate',lr, ...
            'MaxEpochs',maxEpochs, ...
            'MiniBatchSize',bs, ...
            'Shuffle','every-epoch', ...
            'ValidationData',{XValN,YValN}, ...
            'ValidationFrequency',valFrequency, ...
            'ValidationPatience',validationPatience, ...
            'Verbose',false,'Plots','none');

        net = trainNetwork(XTrainN,YTrainN,layers,options);
        pred = double(predict(net,XTestN,'MiniBatchSize',bs));
        pred = pred(:);

        [rmse,mae,r2,smape] = metrics(double(YTestN(:)),pred);

        rr = rr + 1;
        results = add_result(results,rr,testBattery,"LSTM_NoEnergy",seed, ...
            w,h,lr,bs,rmse,mae,r2,smape);

        predictionTable = add_predictions(predictionTable,testBattery, ...
            "LSTM_NoEnergy",seed,testCyclesN,YTestN,pred);
    end

    % ==========================================================
    % C-E) CLASSICAL BENCHMARKS ON SAME TEMPORAL WINDOWS
    % Flatten each [features x window] sequence into one row.
    % ==========================================================
    Xtr = flatten_windows(XTrainA);
    Xte = flatten_windows(XTestA);
    ytr = double(YTrainA(:));
    yte = double(YTestA(:));

    % Random Forest / Bagged Trees — repeated seeds
    for si = 1:numel(seeds)
        seed = seeds(si);
        rng(seed,'twister');

        rf = fitrensemble(Xtr,ytr, ...
            'Method','Bag', ...
            'NumLearningCycles',300, ...
            'Learners',templateTree('MinLeafSize',5));

        pred = predict(rf,Xte);
        [rmse,mae,r2,smape] = metrics(yte,pred);

        rr = rr + 1;
        results = add_result(results,rr,testBattery,"RandomForest",seed, ...
            w,NaN,NaN,NaN,rmse,mae,r2,smape);

        predictionTable = add_predictions(predictionTable,testBattery, ...
            "RandomForest",seed,testCycles,YTestA,pred);
    end

    % Boosted regression trees — repeated seeds
    for si = 1:numel(seeds)
        seed = seeds(si);
        rng(seed,'twister');

        boost = fitrensemble(Xtr,ytr, ...
            'Method','LSBoost', ...
            'NumLearningCycles',300, ...
            'LearnRate',0.05, ...
            'Learners',templateTree('MaxNumSplits',20,'MinLeafSize',5));

        pred = predict(boost,Xte);
        [rmse,mae,r2,smape] = metrics(yte,pred);

        rr = rr + 1;
        results = add_result(results,rr,testBattery,"BoostedTrees",seed, ...
            w,NaN,0.05,NaN,rmse,mae,r2,smape);

        predictionTable = add_predictions(predictionTable,testBattery, ...
            "BoostedTrees",seed,testCycles,YTestA,pred);
    end

    % SVR is deterministic for fixed data; one run is sufficient.
    svr = fitrsvm(Xtr,ytr, ...
        'KernelFunction','gaussian', ...
        'KernelScale','auto', ...
        'Standardize',false);

    pred = predict(svr,Xte);
    [rmse,mae,r2,smape] = metrics(yte,pred);

    rr = rr + 1;
    results = add_result(results,rr,testBattery,"SVR",0, ...
        w,NaN,NaN,NaN,rmse,mae,r2,smape);

    predictionTable = add_predictions(predictionTable,testBattery, ...
        "SVR",0,testCycles,YTestA,pred);

    fprintf('Finished %s\n\n',testBattery);
end

writetable(results,'stage4_benchmark_all_runs.csv');
writetable(predictionTable,'stage4_benchmark_predictions.csv');

% ==============================================================
% SUMMARY PER MODEL / BATTERY
% ==========================================================
models = unique(results.Model,'stable');
summary = table();

for mi = 1:numel(models)
    for bi = 1:numel(batteryNames)
        idx = results.Model == models(mi) & ...
              results.TestBattery == batteryNames(bi);
        R = results(idx,:);

        row = table();
        row.Model = models(mi);
        row.TestBattery = batteryNames(bi);
        row.N_runs = height(R);
        row.RMSE_mean = mean(R.RMSE);
        row.RMSE_SD = std(R.RMSE);
        row.MAE_mean = mean(R.MAE);
        row.MAE_SD = std(R.MAE);
        row.R2_mean = mean(R.R2);
        row.R2_SD = std(R.R2);
        row.sMAPE_mean_percent = mean(R.sMAPE_percent);
        row.sMAPE_SD_percent = std(R.sMAPE_percent);

        summary = [summary; row];
    end
end

writetable(summary,'stage4_benchmark_summary_by_battery.csv');

% ==============================================================
% OVERALL MODEL SUMMARY — BATTERY-LEVEL MACRO AGGREGATION
% ==============================================================
% The held-out battery is the independent cross-battery evaluation unit.
% First average repeated stochastic seeds within each held-out battery,
% then compute the macro mean and SD across the battery-level means.
overall = table();

for mi = 1:numel(models)
    S = summary(summary.Model == models(mi),:);

    row = table();
    row.Model = models(mi);
    row.N_batteries = height(S);
    row.RMSE_mean = mean(S.RMSE_mean);
    row.RMSE_SD = std(S.RMSE_mean);
    row.MAE_mean = mean(S.MAE_mean);
    row.MAE_SD = std(S.MAE_mean);
    row.R2_mean = mean(S.R2_mean);
    row.R2_SD = std(S.R2_mean);
    row.sMAPE_mean_percent = mean(S.sMAPE_mean_percent);
    row.sMAPE_SD_percent = std(S.sMAPE_mean_percent);

    overall = [overall; row];
end

overall = sortrows(overall,'RMSE_mean','ascend');
writetable(overall,'stage4_benchmark_overall_macro_by_battery.csv');

disp('=== OVERALL MODEL COMPARISON: BATTERY-LEVEL MACRO MEAN +/- SD ===');
disp(overall);

fprintf('\nStage 4 complete.\n');


% ==============================================================
% LOCAL FUNCTIONS
% ==============================================================

function layers = make_lstm_layers(numFeatures,hiddenUnits)
layers = [
    sequenceInputLayer(numFeatures,'Name','input')
    lstmLayer(hiddenUnits,'OutputMode','last','Name','lstm')
    fullyConnectedLayer(32,'Name','fc1')
    reluLayer('Name','relu')
    fullyConnectedLayer(1,'Name','output')
    regressionLayer('Name','regression')];
end

function [A,B,C] = scale_three(Araw,Braw,Craw,featureNames)
X = table2array(Araw(:,featureNames));
mu = mean(X,1);
sigma = std(X,0,1);
sigma(sigma < eps) = 1;

A = apply_scaler(Araw,featureNames,mu,sigma);
B = apply_scaler(Braw,featureNames,mu,sigma);
C = apply_scaler(Craw,featureNames,mu,sigma);
end

function Tout = apply_scaler(Tin,featureNames,mu,sigma)
Tout = Tin;
for j = 1:numel(featureNames)
    Tout.(featureNames{j}) = ...
        (Tin.(featureNames{j})-mu(j))/sigma(j);
end
end

function [XSeq,Y,targetCycles] = make_temporal_windows(T,featureNames,w)
batteryNames = unique(T.Battery,'stable');
XSeq = {};
Y = [];
targetCycles = [];

for b = 1:numel(batteryNames)
    B = T(T.Battery == batteryNames(b),:);
    B = sortrows(B,'Cycle');

    X = table2array(B(:,featureNames));
    y = B.SOH;
    cyc = B.Cycle;

    for t = w:(height(B)-1)
        XSeq{end+1,1} = X(t-w+1:t,:)'; %#ok<AGROW>
        Y(end+1,1) = y(t+1); %#ok<AGROW>
        targetCycles(end+1,1) = cyc(t+1); %#ok<AGROW>
    end
end
end

function Xflat = flatten_windows(XSeq)
n = numel(XSeq);
p = numel(XSeq{1});
Xflat = zeros(n,p);
for i = 1:n
    Xflat(i,:) = reshape(XSeq{i},1,[]);
end
end

function [rmse,mae,r2,smape] = metrics(y,pred)
y = double(y(:));
pred = double(pred(:));
rmse = sqrt(mean((pred-y).^2));
mae = mean(abs(pred-y));
r2 = 1 - sum((pred-y).^2)/sum((y-mean(y)).^2);
smape = mean(2*abs(pred-y)./(abs(pred)+abs(y)+eps))*100;
end

function T = add_result(T,row,testBattery,model,seed,w,h,lr,bs,rmse,mae,r2,smape)
T.TestBattery(row) = testBattery;
T.Model(row) = model;
T.Seed(row) = seed;
T.Window(row) = w;
T.HiddenUnits(row) = h;
T.LearningRate(row) = lr;
T.BatchSize(row) = bs;
T.RMSE(row) = rmse;
T.MAE(row) = mae;
T.R2(row) = r2;
T.sMAPE_percent(row) = smape;
end

function P = add_predictions(P,testBattery,model,seed,cycles,y,pred)
Q = table();
n = numel(pred);
Q.TestBattery = repmat(testBattery,n,1);
Q.Model = repmat(model,n,1);
Q.Seed = repmat(seed,n,1);
Q.Cycle = cycles(:);
Q.ActualSOH = double(y(:));
Q.PredictedSOH = double(pred(:));
Q.Residual = Q.PredictedSOH-Q.ActualSOH;
P = [P; Q];
end
