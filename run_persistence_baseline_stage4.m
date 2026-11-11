clc; clear; close all;

% ==============================================================
% PERSISTENCE BASELINE — One-Cycle-Ahead SOH Prediction
% ==============================================================
% Evaluate SOH_hat(t+1) = SOH(t) on the same eligible target cycles
% used by Stage 4 for each outer held-out battery.

load('NASA_corrected_features.mat','All');
cfg = readtable('stage3c_selected_configs.csv');

batteryNames = unique(All.Battery,'stable');

byBattery = table();
predictionTable = table();

for outer = 1:numel(batteryNames)

    testBattery = batteryNames(outer);

    c = cfg(cfg.OuterTestBattery == testBattery,:);
    if isempty(c)
        error('No Stage-3C configuration found for %s.', testBattery);
    end
    c = c(1,:);
    w = c.Window;

    B = All(All.Battery == testBattery,:);
    B = sortrows(B,'Cycle');

    y = double(B.SOH(:));
    cyc = B.Cycle;

    idxPrev   = (w:(height(B)-1))';
    idxTarget = idxPrev + 1;

    actual = y(idxTarget);
    pred   = y(idxPrev);
    targetCycles = cyc(idxTarget);

    [rmse,mae,r2,smape] = metrics(actual,pred);

    row = table();
    row.Model = "Persistence";
    row.TestBattery = testBattery;
    row.Window = w;
    row.N_predictions = numel(actual);
    row.RMSE = rmse;
    row.MAE = mae;
    row.R2 = r2;
    row.sMAPE_percent = smape;
    byBattery = [byBattery; row]; %#ok<AGROW>

    P = table();
    P.TestBattery = repmat(testBattery,numel(actual),1);
    P.Model = repmat("Persistence",numel(actual),1);
    P.Window = repmat(w,numel(actual),1);
    P.Cycle = targetCycles(:);
    P.ActualSOH = actual(:);
    P.PredictedSOH = pred(:);
    P.Residual = pred(:) - actual(:);
    predictionTable = [predictionTable; P]; %#ok<AGROW>

    fprintf('%s | w=%d | N=%d | RMSE=%.6f | MAE=%.6f | R2=%.6f | sMAPE=%.4f%%\n', ...
        char(string(testBattery)), w, numel(actual), rmse, mae, r2, smape);
end

overall = table();
overall.Model = "Persistence";
overall.N_batteries = height(byBattery);
overall.RMSE_mean = mean(byBattery.RMSE);
overall.RMSE_SD = std(byBattery.RMSE);
overall.MAE_mean = mean(byBattery.MAE);
overall.MAE_SD = std(byBattery.MAE);
overall.R2_mean = mean(byBattery.R2);
overall.R2_SD = std(byBattery.R2);
overall.sMAPE_mean_percent = mean(byBattery.sMAPE_percent);
overall.sMAPE_SD_percent = std(byBattery.sMAPE_percent);

writetable(byBattery,'persistence_baseline_by_battery.csv');
writetable(predictionTable,'persistence_baseline_predictions.csv');
writetable(overall,'persistence_baseline_overall_macro_by_battery.csv');

disp(' ');
disp('=== PERSISTENCE BASELINE: BATTERY-LEVEL RESULTS ===');
disp(byBattery);

disp(' ');
disp('=== PERSISTENCE BASELINE: MACRO MEAN +/- BETWEEN-BATTERY SD ===');
disp(overall);

fprintf('\nCreated:\n');
fprintf('  persistence_baseline_by_battery.csv\n');
fprintf('  persistence_baseline_predictions.csv\n');
fprintf('  persistence_baseline_overall_macro_by_battery.csv\n');

function [rmse,mae,r2,smape] = metrics(y,pred)
y = double(y(:));
pred = double(pred(:));
rmse = sqrt(mean((pred-y).^2));
mae = mean(abs(pred-y));
r2 = 1 - sum((pred-y).^2) / sum((y-mean(y)).^2);
smape = mean(2*abs(pred-y)./(abs(pred)+abs(y)+eps))*100;
end
