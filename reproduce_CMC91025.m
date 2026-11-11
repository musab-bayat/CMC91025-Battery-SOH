function reproduce_CMC91025()
% Final-manuscript analysis launcher. Place four NASA .mat files here.
root = fileparts(mfilename('fullpath'));
old = pwd;
restore = onCleanup(@() cd(old));
cd(root);
required = {'B0005.mat','B0006.mat','B0007.mat','B0018.mat'};
for k = 1:numel(required)
    if ~isfile(required{k}), error('Missing file: %s', required{k}); end
end
steps = {'run_preprocessing_stage1.m', ...
 'run_feature_screening_stage2.m', ...
 'run_lstm_stage3c_nested.m', ...
 'run_stage4_energy_ablation_benchmarks_MACRO.m', ...
 'run_stage4_nested_tuned_classical.m', ...
 'run_persistence_baseline_stage4.m', ...
 'run_stage5_rf_importance_energy_ablation_MACRO.m', ...
 'run_stage6_rf_shap_MACRO.m', ...
 'run_multihorizon_evaluation.m', ...
 'run_joint_feature_ablation.m'};
for k = 1:numel(steps)
    fprintf('\n[%d/%d] %s\n', k, numel(steps), steps{k});
    runStage(steps{k});
end
end

function runStage(scriptName)
% Isolate original scripts that start with 'clear' from the launcher.
run(scriptName);
end
