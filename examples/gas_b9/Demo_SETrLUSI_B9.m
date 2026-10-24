function Demo_SETrLUSI_B9(mode)
% Legacy SETrLUSI only, four B9 tasks, seeds 1:10. No spreadsheet input.
% Demo_SETrLUSI_B9('check') checks dependencies/config without fitting.
if nargin == 0, mode = 'run'; end
assert(any(strcmp(mode, {'run','check'})), 'Mode must be run or check.');
here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(here));
support = fullfile(here,'support');
old_path = path;
path_cleanup = onCleanup(@() path(old_path));
addpath(root, fullfile(root,'Model'), ...
    fullfile(root,'Model','models','DL'), fullfile(root,'Model','models','SVM'), ...
    fullfile(root,'Model','models','SVM','liblinear-2.47','matlab'), ...
    fullfile(root,'tools'));
addpath(support,here);
cfg = B9_demo_config();
assert(isequal(cfg.seeds,1:10) && numel(cfg.task_names)==4 && cfg.target==9);
assert(numel(unique(cfg.task_names))==4 && all(cellfun(@(s) ~ismember(9,s),cfg.sources)));
assert(cfg.pa.H==30 && strcmp(cfg.scaleMode,'zscore'));

% Hash code and raw batches so resume cannot silently mix different inputs.
names = {'B9_demo_config','load_gas_sensor','TELUSI_Ram_Multi_F', ...
    'TT','bootstraping','KerF','Vmatrix','DPcaculate','LUSI_VSVM', ...
    'lintrain','linpredict','LIB_train'};
fingerprint = cell(numel(names)+5,2);
for k = 1:numel(names)
    source = which(names{k});
    assert(~isempty(source), 'Missing dependency: %s', names{k});
    assert(startsWith(lower(source),lower([root filesep])), ...
        'Dependency shadowed by another project: %s',source);
    fingerprint(k,:) = {names{k}, file_hash(source)};
end
fingerprint(end-4,:) = {'Demo_SETrLUSI_B9', file_hash(mfilename('fullpath') + ".m")};
datdir = fullfile(here,'data');
for batch = 6:9
    source = fullfile(datdir,sprintf('batch%d.dat',batch));
    assert(isfile(source),'Missing raw data: %s',source);
    fingerprint(end-9+batch,:) = {sprintf('batch%d.dat',batch),file_hash(source)};
end
assert(exist('confusionmat','file')~=0 && exist('perfcurve','file')~=0, ...
    'Statistics and Machine Learning Toolbox is required.');
fprintf('SETrLUSI legacy demo: 4 tasks x 10 seeds, H=30, zscore.\n');
fprintf('Each class searches 4 RD x 3 p3 x 3 p1 configurations.\n');
fprintf('Selection uses held-out labels. Outputs are not independent test estimates.\n');
fprintf('Time includes search. This is not the separate fixed-parameter fit_s.\n');
if strcmp(mode,'check')
    disp(cfg);
    fprintf('Dependency/config check completed. No training or output writes.\n');
    return;
end

outdir = fullfile(here,'results');
if ~isfolder(outdir), mkdir(outdir); end
manifest = fullfile(outdir,'settings.mat');
if isfile(manifest)
    previous = load(manifest);
    assert(isequaln(previous.cfg,cfg) && isequaln(previous.fingerprint,fingerprint), ...
        'Configuration/code/data changed. Use a new output directory, not old checkpoints.');
else
    atomic_mat(manifest,struct('cfg',cfg,'fingerprint',{fingerprint}));
end
all_runs = table();
summary = table();
for ti = 1:numel(cfg.task_names)
    task = cfg.task_names{ti};
    taskdir = fullfile(outdir,task);
    if ~isfolder(taskdir), mkdir(taskdir); end
    runs = table();
    for si = 1:numel(cfg.seeds)
        seed = cfg.seeds(si);
        checkpoint = fullfile(taskdir,sprintf('seed_%02d.mat',seed));
        fprintf('[%s | SETrLUSI] %d/10 phase=load seed=%d\n',task,si-1,seed);
        if isfile(checkpoint)
            saved = load(checkpoint,'record','cfg','fingerprint','task','seed');
            assert(isequaln(saved.cfg,cfg) && isequaln(saved.fingerprint,fingerprint) ...
                && strcmp(saved.task,task) && saved.seed==seed,'Invalid checkpoint.');
            record = saved.record;
            fprintf('  resume: verified saved seed %d\n',seed);
        else
            try
                Data = load_gas_sensor(cfg.sources{ti},cfg.target,cfg.label_ratio,seed,cfg.scaleMode,datdir);
                split.X_train_T = Data.X_train_T;
                split.Y_train_T = Data.Y_train_T;
                split.X_eval_T = Data.X_test_T;
                split.Y_eval_T = Data.Y_test_T;
                split.rng_before_model = rng;
                assert(size(split.X_train_T,1)==50 && size(split.X_eval_T,1)==420, ...
                    'Unexpected B9 target partition size.');
                fprintf('[%s | SETrLUSI] %d/10 phase=search/train/evaluate seed=%d\n',task,si-1,seed);
                timer = tic;
                result = TELUSI_Ram_Multi_F(Data,cfg.ktype,cfg.v_ker,cfg.CDFx,cfg.pa);
                runtime_s = toc(timer);
                assert(isfinite(result.ac_test) && result.ac_test>=0 && result.ac_test<=100);
                assert(abs(mean(result.Ypre_test==split.Y_eval_T)*100-result.ac_test)<1e-9, ...
                    'Predictions and reported accuracy disagree.');
                record = table(string(task),seed,result.ac_test,result.macro_F1, ...
                    result.avg_Sen,runtime_s,string(mat2str(result.best_p1)), ...
                    string(mat2str(result.best_RD)),string(mat2str(result.best_p3)), ...
                    'VariableNames',{'Task','Seed','AC','MacroF1','BAcc', ...
                    'SearchRunSeconds','BestP1Exponent','BestRD','BestP3'});
                fprintf('[%s | SETrLUSI] %d/10 phase=save/verify seed=%d\n',task,si-1,seed);
                atomic_mat(checkpoint,struct('record',record,'result',result, ...
                    'split',split,'cfg',cfg,'fingerprint',{fingerprint},'task',task,'seed',seed));
            catch ME
                fprintf(2,'[%s | SETrLUSI] failed at seed %d: %s\n',task,seed,ME.message);
                rethrow(ME);
            end
        end
        if isempty(runs), runs = record; else, runs = [runs;record]; end %#ok<AGROW>
        atomic_csv(fullfile(taskdir,'per_seed.csv'),runs);
        fprintf('[%s | SETrLUSI] %d/10 phase=verified AC=%.4f%%\n',task,si,record.AC);
    end
    assert(isequal(runs.Seed,cfg.seeds(:)));
    row = table(string(task),height(runs),mean(runs.AC),std(runs.AC,0), ...
        mean(runs.MacroF1),std(runs.MacroF1,0),mean(runs.BAcc),std(runs.BAcc,0), ...
        mean(runs.SearchRunSeconds),'VariableNames',{'Task','N','ACMean','ACStd', ...
        'MacroF1Mean','MacroF1Std','BAccMean','BAccStd','MeanSearchRunSeconds'});
    atomic_csv(fullfile(taskdir,'summary.csv'),row);
    if isempty(all_runs), all_runs = runs; summary = row;
    else, all_runs = [all_runs;runs]; summary = [summary;row]; end %#ok<AGROW>
    atomic_csv(fullfile(outdir,'all_seeds.csv'),all_runs);
    atomic_csv(fullfile(outdir,'summary.csv'),summary);
    fprintf('[%s | SETrLUSI] completed: %.4f +/- %.4f%%\n',task,row.ACMean,row.ACStd);
end
disp(summary);
fprintf('Results: %s\n',outdir);
end

function atomic_mat(destination,payload)
pending = [destination '.pending.mat'];
save(pending,'-struct','payload','-v7');
check = load(pending);
assert(isequaln(check,payload),'MAT read-back failed: %s',pending);
[ok,msg] = movefile(pending,destination,'f');
assert(ok,'%s',msg);
end

function atomic_csv(destination,values)
pending = [destination '.pending.csv'];
writetable(values,pending);
check = readtable(pending,'TextType','string');
assert(height(check)==height(values) && isequal(check.Properties.VariableNames, ...
    values.Properties.VariableNames),'CSV read-back failed: %s',pending);
for k = 1:width(values)
    a = values{:,k}; b = check{:,k};
    if isnumeric(a)
        assert(all(abs(a-b)<=1e-10*max(1,max(abs(a)))), ...
            'CSV numeric read-back failed: %s',pending);
    else
        assert(isequal(string(a),string(b)),'CSV text read-back failed: %s',pending);
    end
end
[ok,msg] = movefile(pending,destination,'f');
assert(ok,'%s',msg);
end

function value = file_hash(filename)
fid = fopen(filename,'rb');
assert(fid>=0,'Cannot read: %s',filename);
cleanup = onCleanup(@() fclose(fid));
bytes = fread(fid,Inf,'*uint8');
digest = java.security.MessageDigest.getInstance('SHA-256');
digest.update(bytes);
value = lower(reshape(dec2hex(typecast(digest.digest(),'uint8'),2).',1,[]));
end
