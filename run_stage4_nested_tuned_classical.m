%% Reviewer 2 Comment 5: nested tuning of classical benchmark models
% Requires NASA_corrected_features.mat and stage3c_selected_configs.csv.
% Preserves Stage 4's outer train/validation/test battery split and
% the LSTM-selected window in each outer fold (common temporal information).
% MATLAB Statistics and Machine Learning Toolbox required.
clc; clear; close all;
load('NASA_corrected_features.mat','All');
cfg = readtable('stage3c_selected_configs.csv');
features = {'V_mean','V_std','V_range','V_slope', ...
    'T_mean','T_std','DeltaT','T_slope','Energy_Wh'};
seeds = [42 123 2026 3407 9999 7 77 314 2718 8080];
innerSeed = 42;
batteries = unique(string(All.Battery),'stable');
assert(numel(batteries)==4,'Expected four batteries.');

% Compact, prespecified model-specific grids; include original settings.
bagGrid = table();
for n = [100 300 500]
    for leaf = [2 5 10]
        bagGrid = [bagGrid; table(n,leaf,'VariableNames',{'NumTrees','Leaf'})]; %#ok<AGROW>
    end
end
boostGrid = table();
for n = [150 300]
    for rate = [0.03 0.05 0.1]
        for splits = [10 20]
            boostGrid = [boostGrid; table(n,rate,splits, ...
                'VariableNames',{'NumTrees','Rate','Splits'})]; %#ok<AGROW>
        end
    end
end
svrGrid = table();
for box = [0.1 1 10]
    for scale = [0.1 1 10]
        for epsilon = [0.001 0.01 0.05]
            svrGrid = [svrGrid; table(box,scale,epsilon, ...
                'VariableNames',{'Box','Scale','Epsilon'})]; %#ok<AGROW>
        end
    end
end
modelNames = ["BaggedTrees","LSBoost","SVR"];
grids = {bagGrid,boostGrid,svrGrid};
selected = table(); innerScores = table(); runs = table(); predictions = table();

for outer = 1:numel(batteries)
    testB = batteries(outer);
    devB = batteries(batteries~=testB);
    c = cfg(string(cfg.OuterTestBattery)==testB,:);
    assert(height(c)==1,'Exactly one Stage 3C config required for %s',testB);
    w = c.Window;
    fprintf('\nOUTER TEST %s | window %d\n',testB,w);

    % Same deterministic final split as Stage 3C/Stage 4.
    counts = zeros(numel(devB),1);
    for k=1:numel(devB), counts(k)=sum(string(All.Battery)==devB(k)); end
    [~,order] = sort(counts,'ascend');
    valB = devB(order(ceil(numel(order)/2)));
    trainB = devB(devB~=valB);

    for mi = 1:numel(modelNames)
        model = modelNames(mi);
        grid = grids{mi};
        scores = inf(height(grid),1);
        for gi = 1:height(grid)
            foldScores = nan(numel(devB),1);
            for iv = 1:numel(devB)
                innerVal = devB(iv);
                innerTrain = devB(devB~=innerVal);
                A = All(ismember(string(All.Battery),innerTrain),:);
                B = All(string(All.Battery)==innerVal,:);
                [A,B] = scale_pair(A,B,features);
                [xa,ya] = windows(A,features,w);
                [xb,yb] = windows(B,features,w);
                try
                    rng(innerSeed,'twister');
                    fit = fit_model(model,xa,ya,grid(gi,:));
                    p = predict(fit,xb);
                    foldScores(iv) = sqrt(mean((p-yb).^2));
                catch ME
                    warning('Inner fit failed: %s outer=%s grid=%d val=%s: %s', ...
                        model,testB,gi,innerVal,ME.message);
                end
            end
            % Do not reward failed configurations with fewer folds.
            if all(isfinite(foldScores)), scores(gi)=mean(foldScores); end
            row = table(testB,model,gi,w,scores(gi), ...
                sum(isfinite(foldScores)), ...
                'VariableNames',{'TestBattery','Model','GridIndex','Window', ...
                'InnerRMSE','ValidInnerFolds'});
            innerScores = [innerScores;row]; %#ok<AGROW>
        end
        [bestRMSE,bestIdx] = min(scores);
        assert(isfinite(bestRMSE),'No valid grid config: %s test=%s',model,testB);
        chosen = grid(bestIdx,:);
        row = table(testB,model,w,valB,bestIdx,bestRMSE, ...
            'VariableNames',{'TestBattery','Model','Window','ValidationBattery', ...
            'GridIndex','InnerRMSE'});
        selected = [selected;row]; %#ok<AGROW>
        fprintf('  %s selected grid=%d innerRMSE=%.5f\n',model,bestIdx,bestRMSE);

        A = All(ismember(string(All.Battery),trainB),:);
        B = All(string(All.Battery)==testB,:);
        [A,B] = scale_pair(A,B,features);
        [xa,ya] = windows(A,features,w);
        [xb,yb,cycles] = windows(B,features,w);
        useSeeds = seeds;
        if model=="SVR",useSeeds=0;end
        for seed=useSeeds
            rng(max(seed,1),'twister');
            fit = fit_model(model,xa,ya,chosen);
            p = double(predict(fit,xb)); p=p(:);
            rmse=sqrt(mean((p-yb).^2)); mae=mean(abs(p-yb));
            r2=1-sum((p-yb).^2)/sum((yb-mean(yb)).^2);
            smape=100*mean(2*abs(p-yb)./(abs(p)+abs(yb)+eps));
            row=table(testB,model,seed,w,bestIdx,rmse,mae,r2,smape, ...
                'VariableNames',{'TestBattery','Model','Seed','Window', ...
                'GridIndex','RMSE','MAE','R2','sMAPE_percent'});
            runs=[runs;row]; %#ok<AGROW>
            n=numel(p);
            P=table(repmat(testB,n,1),repmat(model,n,1), ...
                repmat(seed,n,1),cycles(:),yb(:),p(:), ...
                'VariableNames',{'TestBattery','Model','Seed','Cycle', ...
                'ActualSOH','PredictedSOH'});
            predictions=[predictions;P]; %#ok<AGROW>
        end
    end
end

% Aggregate seeds within each battery; macro mean/SD across FOUR batteries.
byBattery=table(); overall=table();
metricNames={'RMSE','MAE','R2','sMAPE_percent'};
for mi=1:numel(modelNames)
    model=modelNames(mi);
    vals=zeros(numel(batteries),numel(metricNames));
    for bi=1:numel(batteries)
        R=runs(runs.Model==model & runs.TestBattery==batteries(bi),:);
        assert(height(R)>0,'Missing outer battery result');
        for j=1:numel(metricNames)
            vals(bi,j)=mean(R.(metricNames{j}));
        end
        byBattery=[byBattery;table(batteries(bi),model,vals(bi,1), ...
            vals(bi,2),vals(bi,3),vals(bi,4), ...
            'VariableNames',{'TestBattery','Model','RMSE','MAE','R2','sMAPE_percent'})]; %#ok<AGROW>
    end
    overall=[overall;table(model,mean(vals(:,1)),std(vals(:,1)), ...
        mean(vals(:,2)),std(vals(:,2)),mean(vals(:,3)),std(vals(:,3)), ...
        mean(vals(:,4)),std(vals(:,4)), ...
        'VariableNames',{'Model','RMSE_mean','RMSE_SD','MAE_mean', ...
        'MAE_SD','R2_mean','R2_SD','sMAPE_mean_percent','sMAPE_SD_percent'})]; %#ok<AGROW>
end
overall=sortrows(overall,'RMSE_mean');
writetable(innerScores,'r2c5_nested_inner_scores.csv');
writetable(selected,'r2c5_nested_selected.csv');
writetable(runs,'r2c5_nested_outer_runs.csv');
writetable(predictions,'r2c5_nested_predictions.csv');
writetable(byBattery,'r2c5_nested_by_battery.csv');
writetable(overall,'r2c5_nested_overall_macro.csv');
disp('=== NESTED-TUNED CLASSICAL MODELS: BATTERY MACRO RESULTS ===');
disp(overall);

function model=fit_model(name,X,y,c)
    switch name
        case "BaggedTrees"
            model=fitrensemble(X,y,'Method','Bag', ...
                'NumLearningCycles',c.NumTrees, ...
                'Learners',templateTree('MinLeafSize',c.Leaf));
        case "LSBoost"
            model=fitrensemble(X,y,'Method','LSBoost', ...
                'NumLearningCycles',c.NumTrees,'LearnRate',c.Rate, ...
                'Learners',templateTree('MaxNumSplits',c.Splits,'MinLeafSize',5));
        case "SVR"
            model=fitrsvm(X,y,'KernelFunction','gaussian', ...
                'BoxConstraint',c.Box,'KernelScale',c.Scale, ...
                'Epsilon',c.Epsilon,'Standardize',false);
        otherwise
            error('Unknown model');
    end
end
function [A,B]=scale_pair(A,B,features)
    X=table2array(A(:,features));
    mu=mean(X,1);sd=std(X,0,1);sd(sd<eps)=1;
    for j=1:numel(features)
        f=features{j};A.(f)=(A.(f)-mu(j))/sd(j);
        B.(f)=(B.(f)-mu(j))/sd(j);
    end
end
function [X,Y,cycles]=windows(T,features,w)
    X=[];Y=[];cycles=[];
    bats=unique(string(T.Battery),'stable');
    for bi=1:numel(bats)
        B=T(string(T.Battery)==bats(bi),:);B=sortrows(B,'Cycle');
        z=table2array(B(:,features));y=B.SOH;
        for t=w:(height(B)-1)
            % Exactly matches reshape of [features x window] Stage 4 sequences.
            X(end+1,:)=reshape(z(t-w+1:t,:)',1,[]); %#ok<AGROW>
            Y(end+1,1)=y(t+1); %#ok<AGROW>
            cycles(end+1,1)=B.Cycle(t+1); %#ok<AGROW>
        end
    end
end
