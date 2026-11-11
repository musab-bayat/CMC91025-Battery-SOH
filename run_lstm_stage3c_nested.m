clc; clear; close all;

% ==============================================================
% STAGE 3C — Strict Nested Leave-One-Battery-Out LSTM Evaluation
% ==============================================================
% Goal:
%   Remove the Stage-3A/B methodological weakness by ensuring that the
%   held-out TEST battery is NEVER used for hyperparameter selection,
%   normalization, early stopping, or model development.
%
% Outer loop:
%   Hold out one battery as TEST.
%
% Inner model selection:
%   For each candidate configuration, rotate each of the remaining three
%   batteries once as validation and train on the other two.
%   Select the configuration with the lowest MEAN inner-validation RMSE.
%
% Final outer evaluation:
%   Using the selected configuration, train on two development batteries
%   and use the third development battery for early stopping.
%   To avoid choosing this stopping cell using the test battery, the
%   validation cell is selected deterministically as the development
%   battery with the median number of cycles (ties resolved by order).
%   Repeat final training over 10 random seeds.
%
% NOTE:
%   This is intentionally conservative. The test battery remains untouched.
% ==============================================================

clc;
load('NASA_corrected_features.mat','All');

featureNames = { ...
    'V_mean','V_std','V_range','V_slope', ...
    'T_mean','T_std','DeltaT','T_slope','Energy_Wh'};

% Compact search space informed by Stage 3A, but re-selected INSIDE
% every outer fold. The outer test battery is never consulted.
windowCandidates = [10 20];
hiddenCandidates = [32 50 64];
learningRateCandidates = [1e-3 5e-4];
batchCandidates = [16 32];

maxEpochs = 200;
validationPatience = 15;

% One fixed seed for INNER hyperparameter ranking.
innerSeed = 42;

% Ten seeds for OUTER performance/stability reporting.
outerSeeds = [42 123 2026 3407 9999 7 77 314 2718 8080];

batteryNames = unique(All.Battery,'stable');

innerResultsAll = table();
selectedConfigs = table();
outerRuns = table();
outerPredictions = table();

outerRunRow = 0;

fprintf('Starting strict nested leave-one-battery-out evaluation...\n\n');

for outer = 1:numel(batteryNames)

    testBattery = batteryNames(outer);
    devBatteries = batteryNames(batteryNames ~= testBattery);

    fprintf('============================================================\n');
    fprintf('OUTER TEST BATTERY: %s\n',testBattery);
    fprintf('Development batteries: %s, %s, %s\n', ...
        devBatteries(1),devBatteries(2),devBatteries(3));

    % ==========================================================
    % INNER LOOP: hyperparameter selection using ONLY dev cells
    % ==========================================================
    innerTable = table();
    innerRow = 0;

    for w = windowCandidates
        for h = hiddenCandidates
            for lr = learningRateCandidates
                for bs = batchCandidates

                    foldRMSE = nan(numel(devBatteries),1);

                    for iv = 1:numel(devBatteries)

                        innerValBattery = devBatteries(iv);
                        innerTrainBatteries = devBatteries(devBatteries ~= innerValBattery);

                        TrainRaw = All(ismember(All.Battery,innerTrainBatteries),:);
                        ValRaw = All(All.Battery == innerValBattery,:);

                        % Fit normalization ONLY on inner training cells.
                        [TrainScaled,ValScaled] = scale_train_and_apply( ...
                            TrainRaw,ValRaw,featureNames);

                        [XTrain,YTrain,~] = make_temporal_windows( ...
                            TrainScaled,featureNames,w);
                        [XVal,YVal,~] = make_temporal_windows( ...
                            ValScaled,featureNames,w);

                        rng(innerSeed,'twister');

                        layers = make_layers(numel(featureNames),h);
                        valFrequency = max(1,floor(numel(YTrain)/bs));

                        options = trainingOptions('adam', ...
                            'InitialLearnRate',lr, ...
                            'MaxEpochs',maxEpochs, ...
                            'MiniBatchSize',bs, ...
                            'Shuffle','every-epoch', ...
                            'ValidationData',{XVal,YVal}, ...
                            'ValidationFrequency',valFrequency, ...
                            'ValidationPatience',validationPatience, ...
                            'Verbose',false, ...
                            'Plots','none');

                        try
                            net = trainNetwork(XTrain,YTrain,layers,options);
                            pred = predict(net,XVal,'MiniBatchSize',bs);
                            pred = double(pred(:));
                            y = double(YVal(:));
                            foldRMSE(iv) = sqrt(mean((pred-y).^2));
                        catch ME
                            warning('Inner failure: outer=%s val=%s w=%d h=%d lr=%g bs=%d\n%s', ...
                                testBattery,innerValBattery,w,h,lr,bs,ME.message);
                        end
                    end

                    innerRow = innerRow + 1;
                    innerTable.OuterTestBattery(innerRow) = testBattery;
                    innerTable.Window(innerRow) = w;
                    innerTable.HiddenUnits(innerRow) = h;
                    innerTable.LearningRate(innerRow) = lr;
                    innerTable.BatchSize(innerRow) = bs;
                    innerTable.InnerRMSE_mean(innerRow) = mean(foldRMSE,'omitnan');
                    innerTable.InnerRMSE_SD(innerRow) = std(foldRMSE,'omitnan');
                    innerTable.ValidInnerFolds(innerRow) = sum(isfinite(foldRMSE));
                end
            end
        end
    end

    innerTable = sortrows(innerTable, ...
        {'InnerRMSE_mean','InnerRMSE_SD'}, {'ascend','ascend'});

    innerResultsAll = [innerResultsAll; innerTable];

    best = innerTable(1,:);

    selectedConfigs = [selectedConfigs; best];

    fprintf('Selected: window=%d hidden=%d lr=%g batch=%d | inner RMSE %.5f +/- %.5f\n', ...
        best.Window,best.HiddenUnits,best.LearningRate,best.BatchSize, ...
        best.InnerRMSE_mean,best.InnerRMSE_SD);

    % ==========================================================
    % OUTER FINAL TRAINING
    % ==========================================================
    % Deterministic early-stopping validation battery:
    % choose median cycle-count dev battery.
    nCycles = zeros(numel(devBatteries),1);
    for k = 1:numel(devBatteries)
        nCycles(k) = sum(All.Battery == devBatteries(k));
    end
    [~,ord] = sort(nCycles,'ascend');
    valBattery = devBatteries(ord(ceil(numel(ord)/2)));
    trainBatteries = devBatteries(devBatteries ~= valBattery);

    TrainRaw = All(ismember(All.Battery,trainBatteries),:);
    ValRaw = All(All.Battery == valBattery,:);
    TestRaw = All(All.Battery == testBattery,:);

    % Scaling parameters are learned ONLY from outer training cells.
    XTrainRaw = table2array(TrainRaw(:,featureNames));
    mu = mean(XTrainRaw,1);
    sigma = std(XTrainRaw,0,1);
    sigma(sigma < eps) = 1;

    TrainScaled = apply_scaler(TrainRaw,featureNames,mu,sigma);
    ValScaled = apply_scaler(ValRaw,featureNames,mu,sigma);
    TestScaled = apply_scaler(TestRaw,featureNames,mu,sigma);

    [XTrain,YTrain,~] = make_temporal_windows( ...
        TrainScaled,featureNames,best.Window);
    [XVal,YVal,~] = make_temporal_windows( ...
        ValScaled,featureNames,best.Window);
    [XTest,YTest,testCycles] = make_temporal_windows( ...
        TestScaled,featureNames,best.Window);

    for s = 1:numel(outerSeeds)

        seed = outerSeeds(s);
        rng(seed,'twister');

        layers = make_layers(numel(featureNames),best.HiddenUnits);
        valFrequency = max(1,floor(numel(YTrain)/best.BatchSize));

        options = trainingOptions('adam', ...
            'InitialLearnRate',best.LearningRate, ...
            'MaxEpochs',maxEpochs, ...
            'MiniBatchSize',best.BatchSize, ...
            'Shuffle','every-epoch', ...
            'ValidationData',{XVal,YVal}, ...
            'ValidationFrequency',valFrequency, ...
            'ValidationPatience',validationPatience, ...
            'Verbose',false, ...
            'Plots','none');

        try
            net = trainNetwork(XTrain,YTrain,layers,options);

            pred = predict(net,XTest,'MiniBatchSize',best.BatchSize);
            pred = double(pred(:));
            y = double(YTest(:));

            rmse = sqrt(mean((pred-y).^2));
            mae = mean(abs(pred-y));
            r2 = 1 - sum((pred-y).^2)/sum((y-mean(y)).^2);
            smape = mean(2*abs(pred-y)./(abs(pred)+abs(y)+eps))*100;

            outerRunRow = outerRunRow + 1;
            outerRuns.TestBattery(outerRunRow) = testBattery;
            outerRuns.ValidationBattery(outerRunRow) = valBattery;
            outerRuns.TrainBattery1(outerRunRow) = trainBatteries(1);
            outerRuns.TrainBattery2(outerRunRow) = trainBatteries(2);
            outerRuns.Seed(outerRunRow) = seed;
            outerRuns.Window(outerRunRow) = best.Window;
            outerRuns.HiddenUnits(outerRunRow) = best.HiddenUnits;
            outerRuns.LearningRate(outerRunRow) = best.LearningRate;
            outerRuns.BatchSize(outerRunRow) = best.BatchSize;
            outerRuns.RMSE(outerRunRow) = rmse;
            outerRuns.MAE(outerRunRow) = mae;
            outerRuns.R2(outerRunRow) = r2;
            outerRuns.sMAPE_percent(outerRunRow) = smape;

            P = table();
            nPred = numel(pred);
            P.TestBattery = repmat(testBattery,nPred,1);
            P.Seed = repmat(seed,nPred,1);
            P.Cycle = testCycles(:);
            P.ActualSOH = y;
            P.PredictedSOH = pred;
            P.Residual = pred-y;
            outerPredictions = [outerPredictions; P];

            fprintf('  seed=%d -> RMSE %.5f | MAE %.5f | R2 %.4f | sMAPE %.3f%%\n', ...
                seed,rmse,mae,r2,smape);

        catch ME
            warning('Outer failure: test=%s seed=%d\n%s', ...
                testBattery,seed,ME.message);
        end
    end

    fprintf('\n');
end

% ==============================================================
% SAVE RAW RESULTS
% ==============================================================
writetable(innerResultsAll,'stage3c_inner_hyperparameter_results.csv');
writetable(selectedConfigs,'stage3c_selected_configs.csv');
writetable(outerRuns,'stage3c_nested_cross_battery_all_runs.csv');
writetable(outerPredictions,'stage3c_nested_cross_battery_predictions.csv');

% ==============================================================
% PER-BATTERY SUMMARY
% ==============================================================
summary = table();

for b = 1:numel(batteryNames)
    idx = outerRuns.TestBattery == batteryNames(b);
    R = outerRuns(idx,:);

    row = table();
    row.TestBattery = batteryNames(b);
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

writetable(summary,'stage3c_nested_cross_battery_summary.csv');

overall = table();
overall.N_runs = height(outerRuns);
overall.RMSE_mean = mean(outerRuns.RMSE);
overall.RMSE_SD = std(outerRuns.RMSE);
overall.MAE_mean = mean(outerRuns.MAE);
overall.MAE_SD = std(outerRuns.MAE);
overall.R2_mean = mean(outerRuns.R2);
overall.R2_SD = std(outerRuns.R2);
overall.sMAPE_mean_percent = mean(outerRuns.sMAPE_percent);
overall.sMAPE_SD_percent = std(outerRuns.sMAPE_percent);

writetable(overall,'stage3c_nested_cross_battery_overall.csv');

disp('=== SELECTED CONFIGURATIONS ===');
disp(selectedConfigs);

disp('=== STRICT NESTED CROSS-BATTERY SUMMARY ===');
disp(summary);

disp('=== OVERALL ===');
disp(overall);

fprintf('\nStage 3C complete.\n');


% ==============================================================
% LOCAL FUNCTIONS
% ==============================================================

function layers = make_layers(numFeatures,hiddenUnits)
layers = [
    sequenceInputLayer(numFeatures,'Name','input')
    lstmLayer(hiddenUnits,'OutputMode','last','Name','lstm')
    fullyConnectedLayer(32,'Name','fc1')
    reluLayer('Name','relu')
    fullyConnectedLayer(1,'Name','output')
    regressionLayer('Name','regression')];
end

function [TrainScaled,OtherScaled] = scale_train_and_apply( ...
    TrainRaw,OtherRaw,featureNames)

XTrainRaw = table2array(TrainRaw(:,featureNames));
mu = mean(XTrainRaw,1);
sigma = std(XTrainRaw,0,1);
sigma(sigma < eps) = 1;

TrainScaled = apply_scaler(TrainRaw,featureNames,mu,sigma);
OtherScaled = apply_scaler(OtherRaw,featureNames,mu,sigma);
end

function Tout = apply_scaler(Tin,featureNames,mu,sigma)
Tout = Tin;
for j = 1:numel(featureNames)
    name = featureNames{j};
    Tout.(name) = (Tin.(name)-mu(j))/sigma(j);
end
end

function [XSeq,Y,targetCycles] = make_temporal_windows( ...
    T,featureNames,windowLength)

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

    % Cycles [t-window+1,...,t] predict SOH at cycle t+1.
    for t = windowLength:(height(B)-1)
        XSeq{end+1,1} = X(t-windowLength+1:t,:)'; %#ok<AGROW>
        Y(end+1,1) = y(t+1); %#ok<AGROW>
        targetCycles(end+1,1) = cyc(t+1); %#ok<AGROW>
    end
end
end
