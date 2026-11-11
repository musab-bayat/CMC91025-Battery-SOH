clc; clear; close all;

% ==============================================================
% STAGE 5 — Random-Forest Feature Importance + Energy Ablation
% ==============================================================
% Based on Stage 4, Random Forest is the strongest cross-battery model.
%
% This stage:
%   1) repeats strict leave-one-battery-out RF evaluation,
%   2) compares RF with vs. without Energy_Wh,
%   3) computes permutation importance on the UNSEEN test battery,
%   4) aggregates temporal-window importance back to physical features.
%
% IMPORTANT:
% - Same temporal information horizon as Stage 4.
% - Test battery is not used for training or normalization.
% - SOH/Capacity/Cycle/Battery are never predictors.
% - Importance is permutation-based rather than impurity-based, reducing
%   bias toward continuous/high-cardinality variables.
% - Because MATLAB TreeBagger SHAP support differs by release, this script
%   first produces a robust model-agnostic importance analysis. SHAP can be
%   added afterward if the installed MATLAB release supports it.
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
batteryNames = unique(All.Battery,'stable');

runs = table();
importanceRows = table();
predictionRows = table();
rr = 0;

fprintf('Starting Stage 5 RF Energy ablation + permutation importance...\n\n');

for outer = 1:numel(batteryNames)

    testBattery = batteryNames(outer);

    c = cfg(cfg.OuterTestBattery == testBattery,:);
    if isempty(c)
        error('No Stage-3C configuration found for %s.',testBattery);
    end
    w = c.Window(1);

    trainBatteries = batteryNames(batteryNames ~= testBattery);

    TrainRaw = All(ismember(All.Battery,trainBatteries),:);
    TestRaw = All(All.Battery == testBattery,:);

    fprintf('TEST %s | TRAIN other 3 cells | window=%d\n',testBattery,w);

    % ==========================================================
    % ALL FEATURES
    % ==========================================================
    [TrainA,TestA] = scale_train_test(TrainRaw,TestRaw,allFeatures);
    [XTrainSeq,YTrain,~] = make_temporal_windows(TrainA,allFeatures,w);
    [XTestSeq,YTest,testCycles] = make_temporal_windows(TestA,allFeatures,w);

    Xtr = flatten_windows(XTrainSeq);
    Xte = flatten_windows(XTestSeq);
    ytr = double(YTrain(:));
    yte = double(YTest(:));

    % ==========================================================
    % WITHOUT ENERGY
    % ==========================================================
    [TrainN,TestN] = scale_train_test(TrainRaw,TestRaw,noEnergyFeatures);
    [XTrainSeqN,YTrainN,~] = make_temporal_windows(TrainN,noEnergyFeatures,w);
    [XTestSeqN,YTestN,~] = make_temporal_windows(TestN,noEnergyFeatures,w);

    XtrN = flatten_windows(XTrainSeqN);
    XteN = flatten_windows(XTestSeqN);
    ytrN = double(YTrainN(:));
    yteN = double(YTestN(:));

    for si = 1:numel(seeds)

        seed = seeds(si);

        % ---------------- RF ALL FEATURES ----------------
        rng(seed,'twister');
        rf = fitrensemble(Xtr,ytr, ...
            'Method','Bag', ...
            'NumLearningCycles',300, ...
            'Learners',templateTree('MinLeafSize',5));

        pred = predict(rf,Xte);
        [rmse,mae,r2,smape] = metrics(yte,pred);

        rr = rr + 1;
        runs = add_result(runs,rr,testBattery,"RF_All",seed,w, ...
            rmse,mae,r2,smape);

        predictionRows = add_predictions(predictionRows,testBattery, ...
            "RF_All",seed,testCycles,yte,pred);

        % Permutation importance on unseen test data.
        % Baseline metric = RMSE. Positive delta means performance worsens
        % when the variable is permuted => variable is informative.
        baseRMSE = rmse;
        nFeatures = numel(allFeatures);

        for f = 1:nFeatures
            cols = temporal_feature_columns(f,nFeatures,w);

            Xperm = Xte;
            % Permute the entire temporal trajectory of this physical
            % feature across samples, preserving within-window structure.
            rng(seed + 10000*f,'twister');
            order = randperm(size(Xperm,1));
            Xperm(:,cols) = Xperm(order,cols);

            pp = predict(rf,Xperm);
            permRMSE = sqrt(mean((pp-yte).^2));
            delta = permRMSE-baseRMSE;

            row = table();
            row.TestBattery = testBattery;
            row.Seed = seed;
            row.Feature = string(allFeatures{f});
            row.BaselineRMSE = baseRMSE;
            row.PermutedRMSE = permRMSE;
            row.DeltaRMSE = delta;

            importanceRows = [importanceRows; row];
        end

        % ---------------- RF WITHOUT ENERGY ----------------
        rng(seed,'twister');
        rfN = fitrensemble(XtrN,ytrN, ...
            'Method','Bag', ...
            'NumLearningCycles',300, ...
            'Learners',templateTree('MinLeafSize',5));

        predN = predict(rfN,XteN);
        [rmseN,maeN,r2N,smapeN] = metrics(yteN,predN);

        rr = rr + 1;
        runs = add_result(runs,rr,testBattery,"RF_NoEnergy",seed,w, ...
            rmseN,maeN,r2N,smapeN);

        predictionRows = add_predictions(predictionRows,testBattery, ...
            "RF_NoEnergy",seed,testCycles,yteN,predN);
    end

    fprintf('Finished %s\n\n',testBattery);
end

writetable(runs,'stage5_rf_energy_ablation_all_runs.csv');
writetable(importanceRows,'stage5_rf_permutation_importance_raw.csv');
writetable(predictionRows,'stage5_rf_predictions.csv');

% ==============================================================
% ENERGY ABLATION SUMMARY
% ==========================================================
models = unique(runs.Model,'stable');
summary = table();

for mi = 1:numel(models)
    for bi = 1:numel(batteryNames)
        idx = runs.Model == models(mi) & runs.TestBattery == batteryNames(bi);
        R = runs(idx,:);

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

writetable(summary,'stage5_rf_energy_ablation_by_battery.csv');

overall = table();
% Battery-level macro aggregation: first average seeds within each held-out
% battery (already stored in summary), then summarize the battery means.
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
writetable(overall,'stage5_rf_energy_ablation_overall_macro_by_battery.csv');

% ==============================================================
% PERMUTATION IMPORTANCE SUMMARY — BATTERY-LEVEL MACRO AGGREGATION
% ==============================================================
featureNames = unique(importanceRows.Feature,'stable');

% First summarize the 10 stochastic seeds within each held-out battery.
impBattery = table();
for bi = 1:numel(batteryNames)
    for fi = 1:numel(featureNames)
        idx = importanceRows.TestBattery == batteryNames(bi) & ...
              importanceRows.Feature == featureNames(fi);
        R = importanceRows(idx,:);

        row = table();
        row.TestBattery = batteryNames(bi);
        row.Feature = featureNames(fi);
        row.N_seeds = height(R);
        row.DeltaRMSE_mean = mean(R.DeltaRMSE);
        row.DeltaRMSE_SD_seed = std(R.DeltaRMSE);
        row.PositiveFraction_seed = mean(R.DeltaRMSE > 0);

        impBattery = [impBattery; row];
    end
end
writetable(impBattery,'stage5_rf_permutation_importance_by_battery.csv');

% Then give each held-out battery equal weight in the overall importance.
impSummary = table();
for fi = 1:numel(featureNames)
    B = impBattery(impBattery.Feature == featureNames(fi),:);

    row = table();
    row.Feature = featureNames(fi);
    row.N_batteries = height(B);
    row.DeltaRMSE_mean = mean(B.DeltaRMSE_mean);
    row.DeltaRMSE_SD = std(B.DeltaRMSE_mean);
    row.DeltaRMSE_median = median(B.DeltaRMSE_mean);
    row.PositiveBatteryFraction = mean(B.DeltaRMSE_mean > 0);

    impSummary = [impSummary; row];
end

impSummary = sortrows(impSummary,'DeltaRMSE_mean','descend');
writetable(impSummary,'stage5_rf_permutation_importance_summary_macro_by_battery.csv');

disp('=== RF ENERGY ABLATION: OVERALL ===');
disp(overall);

disp('=== RF PERMUTATION IMPORTANCE ===');
disp(impSummary);

fprintf('\nStage 5 complete.\n');


% ==============================================================
% LOCAL FUNCTIONS
% ==============================================================

function [TrainScaled,TestScaled] = scale_train_test( ...
    TrainRaw,TestRaw,featureNames)

X = table2array(TrainRaw(:,featureNames));
mu = mean(X,1);
sigma = std(X,0,1);
sigma(sigma < eps) = 1;

TrainScaled = apply_scaler(TrainRaw,featureNames,mu,sigma);
TestScaled = apply_scaler(TestRaw,featureNames,mu,sigma);
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

function cols = temporal_feature_columns(featureIndex,nFeatures,w)
% MATLAB reshape of [nFeatures x w] places features for time 1 first,
% then features for time 2, etc. Thus one physical feature occupies:
cols = featureIndex:nFeatures:(nFeatures*w);
end

function [rmse,mae,r2,smape] = metrics(y,pred)
y = double(y(:));
pred = double(pred(:));
rmse = sqrt(mean((pred-y).^2));
mae = mean(abs(pred-y));
r2 = 1 - sum((pred-y).^2)/sum((y-mean(y)).^2);
smape = mean(2*abs(pred-y)./(abs(pred)+abs(y)+eps))*100;
end

function T = add_result(T,row,testBattery,model,seed,w,rmse,mae,r2,smape)
T.TestBattery(row) = testBattery;
T.Model(row) = model;
T.Seed(row) = seed;
T.Window(row) = w;
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
