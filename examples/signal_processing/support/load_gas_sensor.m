function Data = load_gas_sensor(source_batches, target_batch, target_label_ratio, seed, scaleMode, datdir)

if nargin < 5 || isempty(scaleMode)
    scaleMode = 'minmax';
end
assert(any(strcmpi(scaleMode, {'zscore','minmax','minmax11'})), ...
    'scaleMode must be ''zscore'', ''minmax'' or ''minmax11''');

if nargin < 6 || isempty(datdir)
    datdir = fullfile(fileparts(mfilename('fullpath')),'..','data');
end

rng(seed);

X_src = cell(1, numel(source_batches));
Y_src = cell(1, numel(source_batches));
for i = 1:numel(source_batches)
    [X_src{i}, Y_src{i}] = parse_batch(datdir, source_batches(i));
end
[X_tgt_all, Y_tgt_all] = parse_batch(datdir, target_batch);

[X_train_T, Y_train_T, X_test_T, Y_test_T] = ...
    TT(X_tgt_all, Y_tgt_all, 1 - target_label_ratio);

X_fit = vertcat(X_src{:}, X_train_T);
switch lower(scaleMode)
    case 'zscore'
        mu = mean(X_fit, 1);
        sd = std(X_fit, 0, 1);
        sd(sd < eps) = 1;
        for i = 1:numel(X_src)
            X_src{i} = (X_src{i} - mu) ./ sd;
        end
        X_train_T = (X_train_T - mu) ./ sd;
        X_test_T  = (X_test_T  - mu) ./ sd;
    case 'minmax'
        lo = min(X_fit, [], 1);
        rg = max(X_fit, [], 1) - lo;
        rg(rg < eps) = 1;
        for i = 1:numel(X_src)
            X_src{i} = (X_src{i} - lo) ./ rg;
        end
        X_train_T = (X_train_T - lo) ./ rg;
        X_test_T  = (X_test_T  - lo) ./ rg;
    case 'minmax11'
        lo = min(X_fit, [], 1);
        rg = max(X_fit, [], 1) - lo;
        rg(rg < eps) = 1;
        for i = 1:numel(X_src)
            X_src{i} = 0.1 + 0.9 * (X_src{i} - lo) ./ rg;
        end
        X_train_T = 0.1 + 0.9 * (X_train_T - lo) ./ rg;
        X_test_T  = 0.1 + 0.9 * (X_test_T  - lo) ./ rg;
end

Data.X_train_A = X_src;
Data.Y_train_A = Y_src;
Data.X_train_T = X_train_T;
Data.Y_train_T = Y_train_T;
Data.X_test_T  = X_test_T;
Data.Y_test_T  = Y_test_T;
end

function [X, Y] = parse_batch(datdir, batch_num)
fpath = fullfile(datdir, sprintf('batch%d.dat', batch_num));
lines = readlines(fpath);
lines = lines(strlength(lines) > 0);
n = numel(lines);
X = zeros(n, 128);
Y = zeros(n, 1);
for i = 1:n
    ln = lines(i);
    sc = strsplit(ln, ';');
    Y(i) = str2double(sc{1});
    rest = sc{2};
    toks = strsplit(strtrim(rest));
    for t = 1:numel(toks)
        kv = strsplit(toks{t}, ':');
        if numel(kv) == 2
            idx = str2double(kv{1});
            val = str2double(kv{2});
            if idx >= 1 && idx <= 128
                X(i, idx) = val;
            end
        end
    end
end
end
