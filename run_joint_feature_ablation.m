% Expanded-training Bagged Trees ablation: nine, eight and seven features.
clc;
load('NASA_corrected_features.mat','All');
cfg=readtable('stage3c_selected_configs.csv');
features={'V_mean','V_std','V_range','V_slope','T_mean','T_std','DeltaT','T_slope','Energy_Wh'};
sets={features,features(~strcmp(features,'Energy_Wh')), ...
    features(~ismember(features,{'Energy_Wh','V_slope'}))};
labels=["All features","Without EnergyWh","Without EnergyWh and Vslope"];
seeds=[42 123 2026 3407 9999 7 77 314 2718 8080];
batteries=unique(All.Battery,'stable');
results=table();predictions=table();
for outer=1:numel(batteries)
    testBattery=batteries(outer);
    c=cfg(cfg.OuterTestBattery==testBattery,:);
    assert(height(c)==1,'Expected one configuration per test battery');
    w=c.Window(1);
    trainRaw=All(All.Battery~=testBattery,:);
    testRaw=All(All.Battery==testBattery,:);
    for k=1:numel(sets)
        f=sets{k}; a=trainRaw;b=testRaw;
        for j=1:numel(f)
            mu=mean(a.(f{j}));sd=std(a.(f{j}));
            if ~isfinite(sd)||sd==0,sd=1;end
            a.(f{j})=(a.(f{j})-mu)/sd;
            b.(f{j})=(b.(f{j})-mu)/sd;
        end
        [Xtr,Ytr,~]=windows(a,f,w);
        [Xte,Yte,cycles]=windows(b,f,w);
        for s=seeds
            rng(s,'twister');
            model=fitrensemble(Xtr,Ytr,'Method','Bag','NumLearningCycles',300, ...
                'Learners',templateTree('MinLeafSize',5));
            p=predict(model,Xte);
            rmse=sqrt(mean((p-Yte).^2));mae=mean(abs(p-Yte));
            r2=1-sum((p-Yte).^2)/sum((Yte-mean(Yte)).^2);
            smape=mean(2*abs(p-Yte)./(abs(p)+abs(Yte)+eps))*100;
            row=table(testBattery,labels(k),s,rmse,mae,r2,smape, ...
                'VariableNames',{'TestBattery','Configuration','Seed','RMSE','MAE','R2','sMAPE_percent'});
            results=[results;row]; %#ok<AGROW>
            Q=table(repmat(testBattery,numel(Yte),1),repmat(labels(k),numel(Yte),1), ...
                repmat(s,numel(Yte),1),cycles(:),Yte(:),p(:), ...
                'VariableNames',{'TestBattery','Configuration','Seed','TargetCycle','TrueSOH','PredictedSOH'});
            predictions=[predictions;Q]; %#ok<AGROW>
        end
    end
end
writetable(results,'joint_ablation_all_runs.csv');
writetable(predictions,'joint_ablation_predictions.csv');
byBattery=groupsummary(results,{'Configuration','TestBattery'},'mean',{'RMSE','MAE','R2','sMAPE_percent'});
writetable(byBattery,'joint_ablation_by_battery.csv');
overall=groupsummary(byBattery,'Configuration',{'mean','std'}, ...
    {'mean_RMSE','mean_MAE','mean_R2','mean_sMAPE_percent'});
writetable(overall,'joint_ablation_macro.csv');
disp(overall);

function [X,Y,cycles]=windows(T,features,w)
X=[];Y=[];cycles=[];
bs=unique(T.Battery,'stable');
for b=1:numel(bs)
    B=sortrows(T(T.Battery==bs(b),:),'Cycle');
    A=table2array(B(:,features));
    for t=w:(height(B)-1)
        win=A(t-w+1:t,:)';
        X(end+1,:)=reshape(win,1,[]); %#ok<AGROW>
        Y(end+1,1)=B.SOH(t+1); %#ok<AGROW>
        cycles(end+1,1)=B.Cycle(t+1); %#ok<AGROW>
    end
end
end
