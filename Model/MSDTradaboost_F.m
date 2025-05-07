function [result] = MSDTradaboost_F(Data,ktype,pa)
% ---------- Data process ----------
X_train_T=Data.X_train_T;
Y_train_T=Data.Y_train_T;
X_test_T=Data.X_test_T;
Y_test_T=Data.Y_test_T;
Y_test_T(Y_test_T==0)=-1;Y_train_T(Y_train_T==0)=-1;
ns=0;
if iscell(Data.X_train_A)
    for Dsnum=1:numel(Data.X_train_A)
        data_ori{Dsnum}.X_train_A=Data.X_train_A{Dsnum};
        data_ori{Dsnum}.Y_train_A=Data.Y_train_A{Dsnum};
        data_ori{Dsnum}.Y_train_A(data_ori{Dsnum}.Y_train_A==0)=-1;
        ns = ns+length(data_ori{Dsnum}.Y_train_A);
    end
else
    data_ori{1}.X_train_A=Data.X_train_A;
    data_ori{1}.Y_train_A=Data.Y_train_A;
    data_ori{1}.Y_train_A(data_ori{1}.Y_train_A==0)=-1;
    ns =length(data_ori{1}.Y_train_A);
end
N=numel(data_ori); % the num of Ds N;
nr = size(Y_train_T, 1);  % Number of samples in target domain
alpha_s = 1 / (1 + sqrt(2 * log(ns) / pa.T));
% Initialize weight vectors
w_s = cell(N, 1);
for k = 1:N
    nk = size(data_ori{k}.X_train_A, 1);
    w_s{k} = ones(nk, 1) / nk;
end
w_r = ones(nr, 1) / nr;
w = [cell2mat(w_s); w_r];
best_error_test=1;
% Main loop
for p1=pa.min:pa.step:pa.max
    fprintf('--------------------*Regular=%.2f*--------------------\n',p1);
    Para.p1 = 2^p1;
    Para.kpar.kp1 =2.^0;
    Para.kpar.kp2 = 0;
    Para.kpar.ktype = "lin";
    % Normalize weight vectors
    for t = 1:pa.T
         w_r = w_r / sum(w_r);e_t = zeros(N, 1);
         ht_weights = zeros(N, 1);ht_T{t}=zeros(size(X_test_T,1),1);
         ht=zeros(size(Y_train_T,1),1);
        for k = 1:N
            w_s{k} = w_s{k} / sum(w_s{k});
            w = [w_s{k}; w_r];
            tX=[data_ori{k}.X_train_A;X_train_T];
            tY=[data_ori{k}.Y_train_A;Y_train_T];
            idx = randsample(size(tX,1), floor(size(tX,1)/2), true, w);
            Trn.X = tX(idx, :);
            Trn.Y = tY(idx);
            [TestPredict{k} , model{k}] = LIB_L1SVC(X_test_T , Trn , Para);
            [TrainPredict{k} , model_train{k}] = LIB_L1SVC(X_train_T , Trn , Para);
            [TrainPredict_A{k} , model_train_A{k}] = LIB_L1SVC(data_ori{k}.X_train_A , Trn , Para);
            ht_A{k}=zeros(size(data_ori{k}.X_train_A,1),1);
            y_pred = TrainPredict{k};
            e_t(k) = sum(w_r .* abs(Y_train_T - y_pred)) / sum(w_r)+10^-8;
            ht_weights(k) = exp(1 - e_t(k)) / exp(e_t(k));
            ht=ht+ht_weights(k)*TrainPredict{k};
            ht_A{k}=ht_A{k}+ht_weights(k)*TrainPredict_A{k};
            ht_T{t}=ht_T{t}+ht_weights(k)*TestPredict{k};
        end
        % Calculate error of candidate classifier on target domain
        y_pred_w = ht./sum(ht_weights);
        epsilon_t = (sum(w_r .*abs(Y_train_T - y_pred_w)) / sum(w_r))+10^-8;
        alpha_t(t) = epsilon_t / (1 - epsilon_t);
        
        for k = 1:N
            w_s{k} = 2*(1-ht_weights(k)).*w_s{k} .* (alpha_s .^ (abs(ht_A{k} - data_ori{k}.Y_train_A)));
        end
        if epsilon_t < 0.5
            w_r = w_r .* (alpha_t(t) .^ (1-epsilon_t));
        end
        label=zeros(size(Y_test_T,1),1);
        DV=zeros(size(Y_test_T,1),1);
        for h=1:t
            label = label + (alpha_t(h).*ht_T{h});
            DV=DV+model{k}.dv;
        end
        label_Final = sign(label);
        CM_test = ConfusionMatrix(label_Final,Y_test_T);
        Test_error(t)=1-CM_test.Ac./100;
        % ---------- Print result ----------
        fprintf('Iteration %s \t',num2str(t))
        fprintf('Train_error=%.4f \t',epsilon_t)
        fprintf('Test_error=%.4f\t\n',Test_error(t))

        folderpath=pa.DA+'/MSD/';
        if t==1
            mkdir(folderpath);
        end
        filename=sprintf('file_%d.mat',p1);
        fullpath=fullfile(folderpath,filename);
        save(fullpath,'Test_error')
    end
    if Test_error(pa.T)<=best_error_test
        best_error_test_H=Test_error;
        best_error_test= Test_error(pa.T);
        best_Final_Y_test=DV;
        best_Ypre_test= label_Final;
        best_regular=p1;
    end
    clear ht_T alpha_t epsilon_t Test_error
end
% ---------- Output ----------
CM_test = ConfusionMatrix(best_Ypre_test,Y_test_T);
result.ac_test=CM_test.Ac;
result.F=CM_test.FM;
result.GM=CM_test.GM;
[~,~,~, AUC]=perfcurve(Y_test_T, sigmoid(best_Final_Y_test./pa.T), '1');
result.AUC=100*AUC;
result.testerror=best_error_test_H;
result.Spe=CM_test.Spe; result.Sen=CM_test.Sen;
result.lam=best_regular;
fprintf('%s\n', repmat('-', 1, 60));
fprintf('Finall_Test_AC=%.4f\t',result.ac_test)
fprintf('Finall_regular=%.4f\t\n',best_regular)
fprintf('%s\n', repmat('=', 1, 60));
end

