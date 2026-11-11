clc; clear; close all;

% ==============================================================
% STAGE 6 — SHAP Analysis for Corrected Random-Forest Model
% ==============================================================
% Prerequisite:
%   Statistics and Machine Learning Toolbox with shapley support.
%
% Design:
%   - Leave one battery completely unseen.
%   - Train RF on the other three batteries.
%   - Training-only normalization.
%   - Same temporal window selected in Stage 3C.
%   - Explain predictions on the unseen test battery.
%   - Flattened temporal-window SHAP values are aggregated back to the
%     nine physical features by summing |SHAP| across all lags.
%
% IMPORTANT:
%   SOH is capacity-based. Energy_Wh is an INPUT feature, not the target.
% ==============================================================

clc;
load('NASA_corrected_features.mat','All');
cfg = readtable('stage3c_selected_configs.csv');

featureNames = { ...
    'V_mean','V_std','V_range','V_slope', ...
    'T_mean','T_std','DeltaT','T_slope','Energy_Wh'};

batteryNames = unique(All.Battery,'stable');

% Fixed RF seed for interpretability. Predictive stability was already
% evaluated across seeds in Stage 5.
rfSeed = 42;

% Limit explained observations if SHAP is computationally expensive.
% Inf = explain all unseen test windows.
maxExplainPerBattery = Inf;

shapRaw = table();
shapSummaryByBattery = table();

fprintf('Starting Stage 6 SHAP analysis...\n\n');

% Check SHAP availability before long computation.
if exist('shapley','file') ~= 2
    error(['The shapley function is not available in this MATLAB release/toolbox. ' ...
           'If this error appears, send me the MATLAB version shown by VER; ' ...
           'we will use the compatible alternative without changing the experiment.']);
end

for outer = 1:numel(batteryNames)

    testBattery = batteryNames(outer);

    c = cfg(cfg.OuterTestBattery == testBattery,:);
    if isempty(c)
        error('No Stage-3C configuration found for %s.',testBattery);
    end
    w = c.Window(1);

    trainBatteries = batteryNames(batteryNames ~= testBattery);

    TrainRaw = All(ismember(All.Battery,trainBatteries),:);
    TestRaw  = All(All.Battery == testBattery,:);

    % Training-only scaling
    [TrainScaled,TestScaled] = scale_train_test( ...
        TrainRaw,TestRaw,featureNames);

    [XTrainSeq,YTrain,~] = make_temporal_windows( ...
        TrainScaled,featureNames,w);
    [XTestSeq,YTest,testCycles] = make_temporal_windows( ...
        TestScaled,featureNames,w);

    Xtr = flatten_windows(XTrainSeq);
    Xte = flatten_windows(XTestSeq);
    ytr = double(YTrain(:));

    rng(rfSeed,'twister');
    rf = fitrensemble(Xtr,ytr, ...
        'Method','Bag', ...
        'NumLearningCycles',300, ...
        'Learners',templateTree('MinLeafSize',5));

    nTest = size(Xte,1);
    if isfinite(maxExplainPerBattery)
        explainIdx = unique(round(linspace(1,nTest, ...
            min(maxExplainPerBattery,nTest))));
    else
        explainIdx = 1:nTest;
    end

    fprintf('TEST %s: explaining %d windows (window=%d)...\n', ...
        testBattery,numel(explainIdx),w);

    % Background/query data for MATLAB shapley.
    explainer = shapley(rf,Xtr);

    localAbs = zeros(numel(explainIdx),numel(featureNames));

    for qi = 1:numel(explainIdx)

        idx = explainIdx(qi);
        q = fit(explainer,Xte(idx,:));

        % MATLAB releases can expose ShapleyValues as a table.
        vals = q.ShapleyValues;
        if istable(vals)
            % Expected value column is generally the second column.
            numericMask = varfun(@isnumeric,vals,'OutputFormat','uniform');
            numericData = table2array(vals(:,numericMask));

            % Prefer last numeric column as SHAP contribution when release
            % adds predictor values plus Shapley values.
            sv = numericData(:,end);
        else
            sv = vals;
        end

        sv = double(sv(:));

        expectedLength = numel(featureNames)*w;
        if numel(sv) ~= expectedLength
            error(['Unexpected SHAP vector length for %s. Expected %d, got %d. ' ...
                   'Send me the MATLAB error/output so the extractor can be ' ...
                   'adapted to your release.'], ...
                   testBattery,expectedLength,numel(sv));
        end

        % Aggregate absolute lag-level SHAP to physical features.
        for f = 1:numel(featureNames)
            cols = temporal_feature_columns(f,numel(featureNames),w);
            localAbs(qi,f) = sum(abs(sv(cols)));

            row = table();
            row.TestBattery = testBattery;
            row.Cycle = testCycles(idx);
            row.Feature = string(featureNames{f});
            row.AbsSHAP_sumAcrossLags = localAbs(qi,f);
            shapRaw = [shapRaw; row];
        end
    end

    for f = 1:numel(featureNames)
        vals = localAbs(:,f);

        row = table();
        row.TestBattery = testBattery;
        row.Feature = string(featureNames{f});
        row.N_explained = numel(vals);
        row.MeanAbsSHAP = mean(vals);
        row.SDAbsSHAP = std(vals);
        row.MedianAbsSHAP = median(vals);

        shapSummaryByBattery = [shapSummaryByBattery; row];
    end

    fprintf('Finished %s\n\n',testBattery);
end

writetable(shapRaw,'stage6_shap_raw.csv');
writetable(shapSummaryByBattery,'stage6_shap_by_battery.csv');

% ==============================================================
% OVERALL SHAP SUMMARY — BATTERY-LEVEL MACRO AGGREGATION
% ==============================================================
% Each held-out battery receives equal weight. Per-battery mean absolute SHAP
% values are treated as the battery-level estimates.
overall = table();

for f = 1:numel(featureNames)
    B = shapSummaryByBattery( ...
        shapSummaryByBattery.Feature == string(featureNames{f}),:);

    row = table();
    row.Feature = string(featureNames{f});
    row.N_batteries = height(B);
    row.MeanAbsSHAP = mean(B.MeanAbsSHAP);
    row.SDAbsSHAP = std(B.MeanAbsSHAP);
    row.MedianAbsSHAP = median(B.MeanAbsSHAP);

    overall = [overall; row];
end

overall = sortrows(overall,'MeanAbsSHAP','descend');
writetable(overall,'stage6_shap_overall_macro_by_battery.csv');

disp('=== CORRECTED RF SHAP IMPORTANCE ===');
disp(overall);

% ==============================================================
% PUBLICATION-QUALITY BAR FIGURE
% ==============================================================
fig = figure('Position',[100 100 900 560]);
barh(categorical(overall.Feature),overall.MeanAbsSHAP);
set(gca,'YDir','reverse');
xlabel('Mean absolute SHAP value (sum across temporal lags)');
ylabel('Physical feature');
title('Random-Forest SHAP Importance under Cross-Battery Evaluation');
grid on;
exportgraphics(fig,'stage6_shap_importance.png','Resolution',300);
exportgraphics(fig,'stage6_shap_importance.pdf','ContentType','vector');

fprintf('\nStage 6 complete.\n');


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
cols = featureIndex:nFeatures:(nFeatures*w);
end
