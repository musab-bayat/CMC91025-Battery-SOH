clc; clear; close all;

% Stage 2: feature screening, redundancy analysis, and leakage checks
load('NASA_corrected_features.mat','All');

% Predictor candidates from Stage 1.
% Current statistics are intentionally excluded from the main candidate set
% because Stage 1 showed they are nearly constant under CC discharge.
features = { ...
    'V_mean','V_std','V_range','V_rms','V_slope', ...
    'T_mean','T_std','DeltaT','T_slope', ...
    'Energy_Wh','DischargeDuration_s'};

target = All.SOH;

%% 1) Missing / non-finite checks and variability
screen = table();

for j = 1:numel(features)
    name = features{j};
    x = All.(name);

    row = table();
    row.Feature = string(name);
    row.N = numel(x);
    row.Missing = sum(ismissing(x));
    row.NonFinite = sum(~isfinite(x));
    row.Mean = mean(x,'omitnan');
    row.Std = std(x,'omitnan');
    row.Min = min(x);
    row.Max = max(x);
    row.Range = row.Max - row.Min;
    row.CV = row.Std / max(abs(row.Mean),eps);
    row.Pearson_r_with_SOH = corr(x,target,'Rows','complete','Type','Pearson');
    row.Spearman_rho_with_SOH = corr(x,target,'Rows','complete','Type','Spearman');

    screen = [screen; row];
end

disp('=== FEATURE SCREENING ===');
disp(screen);
writetable(screen,'stage2_feature_screening.csv');

%% 2) Predictor-predictor correlation matrix
X = zeros(height(All),numel(features));
for j = 1:numel(features)
    X(:,j) = All.(features{j});
end

R = corr(X,'Rows','pairwise','Type','Pearson');
corrTable = array2table(R,'VariableNames',features,'RowNames',features);
writetable(corrTable,'stage2_predictor_correlation.csv','WriteRowNames',true);

%% 3) Highly correlated pairs
threshold = 0.95;
pairRows = table();

for i = 1:numel(features)-1
    for j = i+1:numel(features)
        if abs(R(i,j)) >= threshold
            row = table();
            row.Feature1 = string(features{i});
            row.Feature2 = string(features{j});
            row.Pearson_r = R(i,j);
            pairRows = [pairRows; row];
        end
    end
end

disp('=== HIGHLY CORRELATED PREDICTOR PAIRS (|r| >= 0.95) ===');
disp(pairRows);
writetable(pairRows,'stage2_high_correlation_pairs.csv');

%% 4) Per-battery feature-to-SOH correlations
batteryNames = unique(All.Battery);
perBattery = table();

for b = 1:numel(batteryNames)
    idx = All.Battery == batteryNames(b);

    for j = 1:numel(features)
        x = All.(features{j});
        x = x(idx);
        y = target(idx);

        row = table();
        row.Battery = batteryNames(b);
        row.Feature = string(features{j});
        row.Pearson_r_with_SOH = corr(x,y,'Rows','complete','Type','Pearson');
        row.Spearman_rho_with_SOH = corr(x,y,'Rows','complete','Type','Spearman');

        perBattery = [perBattery; row];
    end
end

writetable(perBattery,'stage2_per_battery_correlations.csv');

%% 5) Explicit Energy diagnostic for Reviewers 12/14/15
energyDiagnostic = table();

for b = 1:numel(batteryNames)
    idx = All.Battery == batteryNames(b);

    row = table();
    row.Battery = batteryNames(b);
    row.Pearson_Energy_SOH = corr(All.Energy_Wh(idx),All.SOH(idx), ...
                                  'Rows','complete','Type','Pearson');
    row.Spearman_Energy_SOH = corr(All.Energy_Wh(idx),All.SOH(idx), ...
                                   'Rows','complete','Type','Spearman');
    row.Pearson_Duration_SOH = corr(All.DischargeDuration_s(idx),All.SOH(idx), ...
                                    'Rows','complete','Type','Pearson');

    energyDiagnostic = [energyDiagnostic; row];
end

disp('=== ENERGY / SOH DIAGNOSTIC ===');
disp(energyDiagnostic);
writetable(energyDiagnostic,'stage2_energy_soh_diagnostic.csv');

%% 6) Current-feature evidence for Reviewer 5
currentFeatures = {'I_mean_abs','I_std','I_rms'};
currentEvidence = table();

for b = 1:numel(batteryNames)
    idx = All.Battery == batteryNames(b);

    for j = 1:numel(currentFeatures)
        name = currentFeatures{j};
        x = All.(name);
        x = x(idx);

        row = table();
        row.Battery = batteryNames(b);
        row.Feature = string(name);
        row.Mean = mean(x,'omitnan');
        row.StdAcrossCycles = std(x,'omitnan');
        row.RangeAcrossCycles = max(x)-min(x);
        row.CV = row.StdAcrossCycles/max(abs(row.Mean),eps);

        currentEvidence = [currentEvidence; row];
    end
end

disp('=== CURRENT FEATURE VARIABILITY (REVIEWER 5) ===');
disp(currentEvidence);
writetable(currentEvidence,'stage2_current_feature_evidence.csv');

%% 7) Suggested manual review list
% IMPORTANT:
% This script does NOT automatically remove a feature merely because it is
% highly correlated with SOH. Strong target correlation is not target leakage
% when the target is independently capacity-based.
%
% Predictor redundancy should be resolved based on BOTH:
%   (a) correlation,
%   (b) physical interpretation.
%
% CapacityCommonAh, NASA_CapacityAh, SOH, Battery and Cycle MUST NOT be
% included as predictors.

fprintf('\nStage 2 complete.\n');
fprintf('Created:\n');
fprintf('  stage2_feature_screening.csv\n');
fprintf('  stage2_predictor_correlation.csv\n');
fprintf('  stage2_high_correlation_pairs.csv\n');
fprintf('  stage2_per_battery_correlations.csv\n');
fprintf('  stage2_energy_soh_diagnostic.csv\n');
fprintf('  stage2_current_feature_evidence.csv\n');
