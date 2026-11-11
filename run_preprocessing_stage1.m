clc; clear; close all;

% Stage 1: corrected preprocessing / feature extraction
files = {'B0005.mat','B0006.mat','B0007.mat','B0018.mat'};

nominalCapacityAh = 2.0;
commonCutoffV = 2.7;
activeCurrentThresholdA = 1.0;

All = table();

for f = 1:numel(files)
    T = build_battery_dataset(files{f}, nominalCapacityAh, ...
                              commonCutoffV, activeCurrentThresholdA);

    fprintf('%s -> %d valid discharge cycles\n', files{f}, height(T));
    All = [All; T];
end

writetable(All, 'NASA_corrected_features.csv');
save('NASA_corrected_features.mat', 'All');

%% Quality-control summary
fprintf('\n=== DATA QUALITY ===\n');
fprintf('Total valid cycles: %d\n', height(All));
fprintf('Missing numeric values: %d\n', ...
    sum(sum(ismissing(All(:, vartype('numeric'))))));

%% Per-battery summary
batteryNames = unique(All.Battery);

summaryRows = table();

for b = 1:numel(batteryNames)
    X = All(All.Battery == batteryNames(b), :);

    tmp = table();
    tmp.Battery = batteryNames(b);
    tmp.N = height(X);
    tmp.SOH_first = X.SOH(1);
    tmp.SOH_last = X.SOH(end);
    tmp.Capacity_first_Ah = X.CapacityCommonAh(1);
    tmp.Capacity_last_Ah = X.CapacityCommonAh(end);
    tmp.Energy_first_Wh = X.Energy_Wh(1);
    tmp.Energy_last_Wh = X.Energy_Wh(end);
    tmp.Imean_SD_across_cycles = std(X.I_mean_abs);
    tmp.Istd_mean = mean(X.I_std);
    tmp.Irms_SD_across_cycles = std(X.I_rms);

    summaryRows = [summaryRows; tmp];
end

disp(summaryRows);
writetable(summaryRows, 'battery_preprocessing_summary.csv');

%% Candidate-feature variability
candidateFeatures = { ...
    'V_mean','V_std','V_range','V_rms','V_slope', ...
    'T_mean','T_std','DeltaT','T_slope', ...
    'Energy_Wh','I_mean_abs','I_std','I_rms'};

featureSummary = table();

for j = 1:numel(candidateFeatures)
    name = candidateFeatures{j};
    x = All.(name);

    row = table();
    row.Feature = string(name);
    row.Mean = mean(x, 'omitnan');
    row.Std = std(x, 'omitnan');
    row.Min = min(x);
    row.Max = max(x);

    denom = max(abs(row.Mean), eps);
    row.CV = row.Std / denom;

    featureSummary = [featureSummary; row];
end

disp(featureSummary);
writetable(featureSummary, 'candidate_feature_variability.csv');

%% Important:
% DO NOT normalize the complete table here.
% DO NOT use CapacityCommonAh, NASA_CapacityAh, SOH, Cycle or Battery as
% predictor features.
%
% Train/validation/test splitting and training-only scaling will be
% implemented in the next stage after this corrected dataset is verified.

fprintf('\nStage 1 complete.\n');
fprintf('Created:\n');
fprintf('  NASA_corrected_features.csv\n');
fprintf('  NASA_corrected_features.mat\n');
fprintf('  battery_preprocessing_summary.csv\n');
fprintf('  candidate_feature_variability.csv\n');
