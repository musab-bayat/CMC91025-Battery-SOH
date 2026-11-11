function T = build_battery_dataset(fileName, nominalCapacityAh, cutoffV, activeCurrentThresholdA)
% BUILD_BATTERY_DATASET
% Leakage-aware preprocessing and feature extraction for NASA PCoE battery data.
%
% Inputs
%   fileName                  e.g. 'B0005.mat'
%   nominalCapacityAh         rated capacity (2.0 Ah for these cells)
%   cutoffV                   common voltage cutoff used for all cells (2.7 V)
%   activeCurrentThresholdA   minimum |I| defining active discharge (1.0 A)
%
% Output
%   T : one row per valid discharge cycle.
%
% Important:
% - Only discharge cycles are used.
% - Post-discharge samples are excluded.
% - All four cells are truncated to a common 2.7 V endpoint.
% - SOH is capacity-based, not energy-based.
% - Capacity, SOH and Cycle are metadata/target columns, NOT predictor features.
% - No normalization is performed here. Scaling must be fitted on training data only.

S = load(fileName);
names = fieldnames(S);
battery = S.(names{1});

rows = struct([]);
r = 0;
dischargeCycle = 0;
fallbackCount = 0;

for k = 1:numel(battery.cycle)
    c = battery.cycle(k);

    if ~strcmpi(c.type, 'discharge')
        continue;
    end

    dischargeCycle = dischargeCycle + 1;
    d = c.data;

    required = {'Voltage_measured','Current_measured','Temperature_measured','Time'};
    if ~all(isfield(d, required))
        continue;
    end

    V = double(d.Voltage_measured(:));
    I = double(d.Current_measured(:));
    Temp = double(d.Temperature_measured(:));
    time = double(d.Time(:));

    % Basic quality checks
    n = numel(V);
    if n < 3 || numel(I) ~= n || numel(Temp) ~= n || numel(time) ~= n
        continue;
    end

    finiteMask = isfinite(V) & isfinite(I) & isfinite(Temp) & isfinite(time);
    V = V(finiteMask);
    I = I(finiteMask);
    Temp = Temp(finiteMask);
    time = time(finiteMask);

    if numel(V) < 3 || any(diff(time) <= 0)
        continue;
    end

    % Identify the beginning of active constant-current discharge.
    % NASA discharge current is approximately -2 A.
    active = abs(I) >= activeCurrentThresholdA;
    firstActive = find(active, 1, 'first');

    if isempty(firstActive)
        continue;
    end

    % Use a common 2.7 V endpoint across B0005/B0006/B0007/B0018.
    % Find the first active-discharge sample at/below cutoff.
    candidateEnd = find((1:numel(V))' >= firstActive & active & V <= cutoffV, ...
                        1, 'first');

    if isempty(candidateEnd)
    % Conservative fallback: last active-discharge sample.
    fallbackCount = fallbackCount + 1;
    candidateEnd = find(active, 1, 'last');
        
    end

    if isempty(candidateEnd) || candidateEnd <= firstActive
        continue;
    end

    idx = firstActive:candidateEnd;

    % Remove any non-active samples inside the selected interval.
    idx = idx(active(idx));

    if numel(idx) < 3
        continue;
    end

    Va = V(idx);
    Ia = I(idx);
    Ta = Temp(idx);
    ta = time(idx);

    % Shift time to zero for numerical clarity.
    ta = ta - ta(1);

    % Capacity to the common cutoff, calculated from measured current.
    capacityCommonAh = trapz(ta, abs(Ia)) / 3600;

    % Capacity-based SOH. A common rated denominator avoids using
    % information from the held-out test cell to normalize its target.
    SOH = capacityCommonAh / nominalCapacityAh;

    % Energy is an INPUT CANDIDATE only; it no longer defines SOH.
    powerW = Va .* abs(Ia);
    energyWh = trapz(ta, powerW) / 3600;

    % Voltage trend over the active discharge segment.
    pV = polyfit(ta, Va, 1);

    % Temperature trend.
    pT = polyfit(ta, Ta, 1);

    r = r + 1;
    rows(r).Battery = string(names{1});
    rows(r).Cycle = dischargeCycle;

    % Target / reference quantities
    rows(r).CapacityCommonAh = capacityCommonAh;
    if isfield(d,'Capacity')
        rows(r).NASA_CapacityAh = double(d.Capacity);
    else
        rows(r).NASA_CapacityAh = NaN;
    end
    rows(r).SOH = SOH;

    % Candidate predictor features
    rows(r).V_mean = mean(Va);
    rows(r).V_std = std(Va);
    rows(r).V_range = max(Va) - min(Va);
    rows(r).V_rms = rms(Va);
    rows(r).V_slope = pV(1);

    rows(r).T_mean = mean(Ta);
    rows(r).T_std = std(Ta);
    rows(r).DeltaT = max(Ta) - min(Ta);
    rows(r).T_slope = pT(1);

    rows(r).Energy_Wh = energyWh;

    % Current statistics are retained temporarily for the reviewer's
    % variance check. They should not automatically be used as predictors.
    rows(r).I_mean_abs = mean(abs(Ia));
    rows(r).I_std = std(Ia);
    rows(r).I_rms = rms(Ia);

    % Diagnostic metadata
    rows(r).DischargeDuration_s = ta(end);
    rows(r).N_active_samples = numel(idx);
    rows(r).V_start = Va(1);
    rows(r).V_end = Va(end);
end

if isempty(rows)
    T = table();
else
    T = struct2table(rows);
end
fprintf('%s -> fallback cycles (did not reach %.1f V): %d\n', ...
        fileName, cutoffV, fallbackCount);

end
