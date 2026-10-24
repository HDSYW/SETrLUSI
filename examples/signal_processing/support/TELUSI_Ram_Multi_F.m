function result = TELUSI_Ram_Multi_F(Data, ktype, v_ker, CDFx, pa)
    if ~isfield(pa, 'thr'), pa.thr = 0.5; end
    if ~isfield(pa, 'reg'), pa.reg = 0; end
    all_labels = unique([Data.Y_train_T; Data.Y_test_T]);
    K = numel(all_labels);
    Final_scores = zeros(size(Data.X_test_T, 1), K);

    for c = 1:K
        Y_train_binary = -ones(size(Data.Y_train_T));
        Y_train_binary(Data.Y_train_T == all_labels(c)) = 1;
        Y_test_binary = -ones(size(Data.Y_test_T));
        Y_test_binary(Data.Y_test_T == all_labels(c)) = 1;

        orig_Y_train_T = Data.Y_train_T;
        orig_Y_test_T = Data.Y_test_T;
        Data.Y_train_T = Y_train_binary;
        Data.Y_test_T = Y_test_binary;

        best_error_test = 1;
        best_error_test_H = [];
        best_Final_Y_test = [];
        best_Ypre_test = [];
        best_minBeta = [];
        best_regular = [];
        best_p3 = [];
        best_RD = [];

        X_train_T = Data.X_train_T; Y_train_T = Data.Y_train_T;
        X_test_T = Data.X_test_T; Y_test_T = Data.Y_test_T;

        if iscell(Data.X_train_A)
            for Dsnum = 1:numel(Data.X_train_A)
                data_ori{Dsnum}.X_train_A = Data.X_train_A{Dsnum};
                data_ori{Dsnum}.Y_train_A = Data.Y_train_A{Dsnum};
            end
        else
            data_ori{1}.X_train_A = Data.X_train_A;
            data_ori{1}.Y_train_A = Data.Y_train_A;
        end

        if ~isfield(pa, 'RDlist'), pa.RDlist = 0.3:0.2:0.9; end
        if ~isfield(pa, 'p3list'), pa.p3list = 0.1:0.4:0.9; end
        fprintf('*Vtype=%s* *Ptype=%s*\n', pa.Vtype, strjoin(cellstr(pa.predicate_ori), ' | '));
        for RD = pa.RDlist
            pa.randomratio = RD;
            for p3 = pa.p3list
                for p1 = pa.min:pa.step:pa.max
                    Para.kpar.kp1 = 2.^0;
                    Para.p1 = 2.^p1;
                    Para.kpar.kp2 = 0;
                    Para.v_sig = 2.^0;
                    Para.p3 = p3;

                    V_Matrix = 'Vmatrix';
                    V_MatrixFun = str2func(V_Matrix);
                    Para.vmatrix = V_Matrix;
                    Para.CDFx = CDFx;
                    Para.v_ker = v_ker;
                    CDFx = Para.CDFx;
                    Para.kpar.ktype = ktype;
                    Para.KP.ktype = Para.kpar.ktype;
                    Para.KP.kp1 = Para.kpar.kp1;
                    Para_svm.kpar.ktype = ktype;
                    Para_svm.kpar.kp1 = 2.^-8;
                    Para_svm.kpar.kp2 = 0;

                    Para.Vtype = pa.Vtype;
                    Para.thr = pa.thr;
                    Para.reg = pa.reg;
                    predicate_ori = pa.predicate_ori;

                    N = numel(data_ori);
                    U = length(predicate_ori);

                    Dt_train = [X_train_T, Y_train_T];

                    w_t = cell(1, N);
                    for i = 1:N
                        w_t{i} = ones(size(Y_train_T, 1), 1);
                    end
                    for h = 1:pa.H
                        Final_Y_test = zeros(size(X_test_T, 1), 1);
                        e_kr = zeros(1, N);
                        for i = 1:N
                            index{i} = randi(U);

                            if contains(pa.ram, 'Open P')
                                Ds_new{i} = [data_ori{i}.X_train_A, data_ori{i}.Y_train_A];
                                Dt_train_new{i} = Dt_train;
                                P{i} = rand(size(Dt_train_new{i}, 1), 1);
                            elseif contains(pa.ram, 'Open DSDT')
                                [~, ~, Ds.X, Ds.Y] = TT(data_ori{i}.X_train_A, data_ori{i}.Y_train_A, pa.randomratio);
                                Ds_new{i} = [Ds.X, Ds.Y];
                                Dt_train_new{i} = bootstraping(Dt_train, size(Y_train_T, 1));
                                P{i} = rand(size(Dt_train_new{i}, 1), 1);
                            elseif contains(pa.ram, 'All Close')
                                Ds_new{i} = [data_ori{i}.X_train_A, data_ori{i}.Y_train_A];
                                Dt_train_new{i} = Dt_train;
                                P{i} = w_t{i} ./ sum(w_t{i});
                            elseif contains(pa.ram, 'All Open')
                                [~, ~, Ds.X, Ds.Y] = TT(data_ori{i}.X_train_A, data_ori{i}.Y_train_A, pa.randomratio);
                                Ds_new{i} = [Ds.X, Ds.Y];
                                Dt_train_new{i} = bootstraping(Dt_train, size(Y_train_T, 1));
                                P{i} = rand(size(Dt_train_new{i}, 1), 1);
                            end

                            idx = randsample(size(Dt_train_new{i}, 1), floor(size(Dt_train_new{i}, 1)/2), true, P{i});
                            Data_train.X = Dt_train_new{i}(idx, 1:end-1);
                            Data_train.Y = Dt_train_new{i}(idx, end);
                            DS.X = Ds_new{i}(:, 1:end-1);
                            DS.Y = Ds_new{i}(:, end);

                            Para_svm.p1 = 2.^(randi([-8, 8]));

                            if any(strcmp(predicate_ori(index{i}), {'A_Y', 'DV'}))
                                model = lintrain(DS.Y, sparse(DS.X), '-q');
                                [Para.A_Y, ~, Para.DV] = linpredict(Data_train.Y, sparse(Data_train.X), model,'-q');
                            else
                                Para.A_Y = 1;
                                Para.DV = 0;
                            end

                            X_train_T_new = Dt_train_new{i}(:, 1:end-1);
                            Y_train_T_new = Dt_train_new{i}(:, end);

                            if contains(Para.Vtype, 'V')
                                [Para.V, ~] = Vmatrix(Data_train.X, CDFx, Para.v_sig, Para.v_ker);
                            else
                                Para.V = eye(size(Data_train.X, 1));
                            end

                            [Para.P, ~] = DPcaculate(Data_train.X, Data_train.Y, Ds_new{i}(:, 1:end-1), ...
                                Para.A_Y, Ds_new{i}(:, end), Para.DV, Para.KP, predicate_ori{index{i}}, Para.p3);
                            if contains(pa.ram, {'All Close', 'Open DSDT'})
                                Para.p3 = 0;
                            end

                            [Y, Model] = LUSI_VSVM(X_test_T, Data_train, Para);
                            [Y_valx, ~] = LUSI_VSVM(X_train_T_new, Data_train, Para);

                            error = mean(Y_valx.tst ~= Y_train_T_new);
                            if error < 0.50
                                e_kr(i) = error;
                            else
                                e_kr(i) = 0.499;
                            end

                            if error == 0
                                e_kr(i) = 0.001;
                            end

                            beta_kt(i) = 1 - e_kr(i) / (1 - e_kr(i));
                            for m = 1:size(Y_train_T_new, 1)
                                w_t{i}(m) = w_t{i}(m) * (beta_kt(i)^(-abs(Y_valx.tst(m) - Y_train_T_new(m))));
                            end

                            beta_Y_test{i, h} = beta_kt(i) * Model.f;
                        end

                        [~, ErIndex(h)] = min(e_kr);
                        minBeta(h) = beta_kt(ErIndex(h));

                        if min(e_kr) == 0.499
                            beta_Y_test{ErIndex(h), h} = zeros(size(Model.f, 1), 1);
                            minBeta(h) = 0;
                        end

                        minpredicate{h} = predicate_ori{index{ErIndex(h)}};

                        for ii = 1:h
                            Final_Y_test = Final_Y_test + beta_Y_test{ErIndex(ii), ii};
                        end

                        Final_Y_P{h}=Final_Y_test;
                        Ypre_test{h} = (sign(Final_Y_P{h} / sum(minBeta) - pa.thr) + 1) / 2;
                        Ypre_test{h}(Ypre_test{h}==0)=-1;
                        final_error_test(h) = mean(abs(Ypre_test{h} - Y_test_T));
                    end

                    [Vmin,idx]=min(final_error_test);
                    if Vmin <= best_error_test
                        best_error_test_H = final_error_test;
                        best_error_test = Vmin;
                        best_Final_Y_test = Final_Y_P{idx};
                        best_regular = p1;
                        best_p3 = Para.p3;
                        best_RD = RD;
                    end

                    if isfield(pa, 'DA')
                        folderpath = num2str(pa.DA)+ "/TE/";
                        mkdir(folderpath);
                        filename = sprintf('file_%d.mat', best_regular);
                        fullpath = fullfile(folderpath, filename);
                        save(fullpath, 'best_error_test_H');
                    end
                end
            end
        end
        Final_scores(:, c) = best_Final_Y_test;
        best_cfg_p1(c) = best_regular;
        best_cfg_RD(c) = best_RD;
        best_cfg_p3(c) = best_p3;

        Data.Y_train_T = orig_Y_train_T;
        Data.Y_test_T = orig_Y_test_T;
    end

    [~, max_idx] = max(Final_scores, [], 2);
    Ypre_test = all_labels(max_idx);

    C = confusionmat(orig_Y_test_T, Ypre_test);

    result.Sen = zeros(1, K);
    for k = 1:K
        TP = C(k, k);
        FN = sum(C(k, :)) - TP;
        result.Sen(k) = TP / (TP + FN);
    end
    result.avg_Sen = mean(result.Sen) * 100;

    result.ac_test = sum(diag(C)) / sum(C(:)) * 100;

    result.F1 = zeros(1, K);
    for k = 1:K
        TP = C(k, k);
        FP = sum(C(:, k)) - TP;
        FN = sum(C(k, :)) - TP;
        prec = TP / max(TP + FP, eps);
        rec  = TP / max(TP + FN, eps);
        result.F1(k) = 2 * prec * rec / max(prec + rec, eps);
    end
    result.macro_F1 = mean(result.F1) * 100;

    result.Ypre_test = Ypre_test;
    result.confusion = C;
    result.Final_scores = Final_scores;
    result.all_labels = all_labels;
    result.best_p1 = best_cfg_p1(:)';
    result.best_RD = best_cfg_RD(:)';
    result.best_p3 = best_cfg_p3(:)';

    aucs = zeros(1, K);
    for k = 1:K
        binary_labels = -ones(size(orig_Y_test_T));
        binary_labels(orig_Y_test_T == all_labels(k)) = 1;
        [~, ~, ~, aucs(k)] = perfcurve(binary_labels, Final_scores(:, k), 1);
    end
    if isfield(pa, 'DA'), pa.DA=pa.DA+1; end
    result.F = NaN;
    result.GM = NaN;
    result.Spe = NaN;
    result.lam = best_regular;
    result.p3 = best_p3;
    result.testerror=best_error_test_H;

    fprintf('Overall accuracy: %.2f%%\n', result.ac_test);
end
