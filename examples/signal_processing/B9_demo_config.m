function cfg = B9_demo_config()
% Historical B9 protocol confirmed by the user. No independent test split.
cfg.protocol = 'legacy_eval_selected_v1';
cfg.method = 'SETrLUSI';
cfg.task_names = {'B6_B9','B6B7_B9','B7B8_B9','B6B7B8_B9'};
cfg.sources = {6, [6 7], [7 8], [6 7 8]};
cfg.target = 9;
cfg.label_ratio = 0.1;
cfg.seeds = 1:10;
cfg.scaleMode = 'zscore';
cfg.ktype = 'lin';
cfg.v_ker = 'theta_ker';
cfg.CDFx = 'uniform';
cfg.pa.H = 30;
cfg.pa.T = 30;
cfg.pa.min = -8;
cfg.pa.step = 4;
cfg.pa.max = 0;
cfg.pa.RDlist = 0.3:0.2:0.9;
cfg.pa.p3list = 0.1:0.4:0.9;
cfg.pa.therthold = 0.5;
cfg.pa.randomratio = 0.9;
cfg.pa.p3 = 0.5;
cfg.pa.ram = "All Open";
cfg.pa.Vtype = "I";
cfg.pa.thr = 0.5;
cfg.pa.reg = 0;
cfg.pa.predicate_ori = ["one","Zero","DV","A_Y","feature", ...
    "Kernel","X","XX","Ram_X","Ram_XX"];
% The legacy model draws the auxiliary SVM exponent with randi([-8,8]).
% lintrain/LIB_train retain their existing arguments in the original code.
cfg.selection = 'per-class evaluation error, including best ensemble round';
cfg.time_scope = 'model call including parameter search, prediction and evaluation';
end
