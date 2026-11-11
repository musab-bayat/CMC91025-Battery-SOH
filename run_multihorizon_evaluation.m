% Multi-horizon sensitivity: 1, 5, 10, 20 cycles ahead.
% Fixed 20-cycle target eligibility, same endpoints for all horizons.
clc;
load('NASA_corrected_features.mat','All');
cfg = readtable('stage3c_selected_configs.csv');
features = {'V_mean','V_std','V_range','V_slope','T_mean','T_std','DeltaT','T_slope','Energy_Wh'};
seeds = [42 123 2026 3407 9999 7 77 314 2718 8080];
horizons = [1 5 10 20];
batteries = unique(All.Battery,'stable');
results = table(); preds = table();
for outer = 1:numel(batteries)
    testBattery = batteries(outer);
    dev = batteries(batteries ~= testBattery);
    c = cfg(cfg.OuterTestBattery == testBattery,:);
    assert(height(c)==1,'Expected one selected configuration per test battery');
    w = c.Window(1);
    counts = zeros(numel(dev),1);
    for j=1:numel(dev), counts(j)=sum(All.Battery==dev(j)); end
    [~,idx] = sort(counts,'ascend');
    valBattery = dev(idx(ceil(numel(dev)/2)));
    trainBatteries = dev(dev~=valBattery);
    trRaw=All(ismember(All.Battery,trainBatteries),:);
    teRaw=All(All.Battery==testBattery,:);
    [tr,te]=scale_pair(trRaw,teRaw,features);
    for h=horizons
        [Xtr,Ytr,~]=horizon_windows(tr,features,w,h,20);
        [Xte,Yte,cycles,priorSOH]=horizon_windows(te,features,w,h,20);
        assert(~isempty(Ytr)&&~isempty(Yte),'Insufficient eligible windows');
        for si=1:numel(seeds)
            seed=seeds(si);
            rng(seed,'twister');
            bag=fitrensemble(Xtr,Ytr,'Method','Bag','NumLearningCycles',300, ...
                'Learners',templateTree('MinLeafSize',5));
            p=predict(bag,Xte);
            [results,preds]=append_result(results,preds,testBattery,"Bagged Trees",h,seed,cycles,Yte,p);
            rng(seed,'twister');
            boost=fitrensemble(Xtr,Ytr,'Method','LSBoost','NumLearningCycles',300, ...
                'LearnRate',0.05,'Learners',templateTree('MaxNumSplits',20,'MinLeafSize',5));
            p=predict(boost,Xte);
            [results,preds]=append_result(results,preds,testBattery,"LSBoost",h,seed,cycles,Yte,p);
        end
        [results,preds]=append_result(results,preds,testBattery,"Persistence",h,NaN,cycles,Yte,priorSOH);
    end
end
writetable(results,'multihorizon_all_runs.csv');
writetable(preds,'multihorizon_predictions.csv');
byBattery=groupsummary(results,{'Model','Horizon','TestBattery'},'mean',{'RMSE','MAE','R2'});
% Each battery contributes equally. Persistence has one result per battery.
writetable(byBattery,'multihorizon_by_battery.csv');
overall=groupsummary(byBattery,{'Model','Horizon'},{'mean','std'}, ...
    {'mean_RMSE','mean_MAE','mean_R2'});
writetable(overall,'multihorizon_macro.csv');
disp(overall);

function [a,b]=scale_pair(a,b,features)
for j=1:numel(features)
    f=features{j}; mu=mean(a.(f)); sd=std(a.(f));
    if ~isfinite(sd)||sd==0, sd=1; end
    a.(f)=(a.(f)-mu)/sd; b.(f)=(b.(f)-mu)/sd;
end
end
function [X,Y,cycles,prior]=horizon_windows(T,features,w,h,maxH)
X=[];Y=[];cycles=[];prior=[];
bs=unique(T.Battery,'stable');
for b=1:numel(bs)
    B=sortrows(T(T.Battery==bs(b),:),'Cycle');
    A=table2array(B(:,features)); y=B.SOH;
    for t=w:(height(B)-maxH)
        win=A(t-w+1:t,:)';
        X(end+1,:)=reshape(win,1,[]); %#ok<AGROW>
        Y(end+1,1)=y(t+h); %#ok<AGROW>
        cycles(end+1,1)=B.Cycle(t+h); %#ok<AGROW>
        prior(end+1,1)=y(t); %#ok<AGROW>
    end
end
end
function [R,P]=append_result(R,P,battery,model,h,seed,cycles,y,p)
y=double(y(:));p=double(p(:));
row=table(battery,model,h,seed,sqrt(mean((p-y).^2)),mean(abs(p-y)), ...
    1-sum((p-y).^2)/sum((y-mean(y)).^2), ...
    'VariableNames',{'TestBattery','Model','Horizon','Seed','RMSE','MAE','R2'});
R=[R;row];
Q=table(repmat(battery,numel(y),1),repmat(model,numel(y),1), ...
    repmat(h,numel(y),1),repmat(seed,numel(y),1),cycles(:),y,p, ...
    'VariableNames',{'TestBattery','Model','Horizon','Seed','TargetCycle','TrueSOH','PredictedSOH'});
P=[P;Q];
end
