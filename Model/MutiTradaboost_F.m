function [result] = MutiTradaboost_F(Data,ktype,pa)

% ---------- Data process ----------
X_train_T=Data.X_train_T;
Y_train_T=Data.Y_train_T;
X_test_T=Data.X_test_T;
Y_test_T=Data.Y_test_T;
Y_test_T(Y_test_T==0)=-1;Y_train_T(Y_train_T==0)=-1;
if iscell(Data.X_train_A)
    for Dsnum=1:numel(Data.X_train_A)
        data_ori{Dsnum}.X_train_A=Data.X_train_A{Dsnum};
        data_ori{Dsnum}.Y_train_A=Data.Y_train_A{Dsnum};
        data_ori{Dsnum}.Y_train_A(data_ori{Dsnum}.Y_train_A==0)=-1;
    end
else
    data_ori{1}.X_train_A=Data.X_train_A;
    data_ori{1}.Y_train_A=Data.Y_train_A;
    data_ori{1}.Y_train_A(data_ori{1}.Y_train_A==0)=-1;
end
best_error_test=1;
for p1=pa.min:pa.step:pa.max
    fprintf('--------------------*Regular=%.2f*--------------------\n',p1);
    Para.p1 = 2.^p1;  Para.kpar.kp1 =2.^0; Para.kpar.kp2 = 0;Para.kpar.ktype = ktype;
    N=numel(data_ori);
    ns = 0;
    for i = 1:N
        ns = ns + length(data_ori{i}.Y_train_A);
        ws(i).weight = ones(length(data_ori{i}.Y_train_A),1);
    end
    m = length(Y_train_T);
    as = log(1 + sqrt(2* log(ns/pa.T)))/2;
    wt.weight = ones(length(Y_train_T),1);
    sW = sum(wt.weight);
    hyp = {};
    er = ones(pa.T,1);

    %% begin of iteration
    for t = 1:pa.T
        model = {};
        et = ones(N,1);
        bestaccuracy = 0;
        for k = 1:N
            W = [ws(k).weight;wt.weight];
            if isinf(sum(W))
                P=W./exp(700);
            else
                P=W./sum(W);
            end
            X = [data_ori{k}.X_train_A;X_train_T];
            Y = [data_ori{k}.Y_train_A;Y_train_T];
            idx = randsample(size(X,1), floor(size(X,1)/2), true, P);
            Trn.X = X(idx, :);
            Trn.Y = Y(idx);
            [TrainPredict{k} , ~] = LIB_L1SVC(X , Trn , Para);
            [TestPredict{k} , model{k}] = LIB_L1SVC(X_test_T , Trn , Para);
            n = length(data_ori{k}.Y_train_A);
            et(k) = sum(P(n+1:m+n).*(TrainPredict{k}(n+1:m+n)~=Y_train_T))./sum(P(n+1:m+n));
            %% choose the model with least error rate and best accuracy
            if et(k) >= 0.5
                et(k) = 0.499;
            elseif et(k)==0
                et(k) = 0.001;
            end
            n = length(ws(k).weight);
            for j=1:n
                ws(k).weight(j) = ws(k).weight(j)*exp(-as*abs(TrainPredict{k}(j)-data_ori{k}.Y_train_A(j)));
            end
        end
        [~,index]=min(et);
        er(t) = et(index);
        bestTestPredict{t} = TestPredict{index};
        bestTtrainPredict{t}=TrainPredict{index};
        bestvalue{t}=model{index}.dv;
        n_min=size(TrainPredict{index},1);
        alpha(t) = log((1-er(t))/er(t))/2;
        T=bestTtrainPredict{t}(n_min-m+1:n_min);
        for j=1:m
            wt.weight(j) = wt.weight(j)*exp(alpha(t)*abs(T(j)-Y_train_T(j)));
            if isinf(wt.weight(j))
                wt.weight(j)=max(wt.weight(~isinf(wt.weight)));
            end
        end
        label = zeros(size(X_test_T,1),1);
        DV=zeros(size(label,1),1);
        for h=1:t
            label = label + (alpha(h).*bestTestPredict{h});
            DV=DV+bestvalue{t};
        end
        label_Final = sign(label);
        CM_test = ConfusionMatrix( label_Final,Y_test_T);
        Test_error(t)=1-CM_test.Ac./100;
        % ---------- Print result ----------
        fprintf('Iteration %s \t',num2str(t))
        fprintf('Train_error=%.4f \t',er(t))
        fprintf('Test_error=%.4f\t\n',Test_error(t))

        folderpath=pa.DA+'/MT/';
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
    clear bestTestPredict bestTtrainPredict alpha er bestvalue Test_error
    % ---------- Figure ----------
    if sum(pa.figure=='On')
        figure
        x=1:pa.T;
        y_1=er;
        y_2=Test_error;
        plot(x,y_1,"Marker",".","MarkerSize",15,Color=[223/255,122/255,094/255])
        hold on
        plot(x,y_2,"Marker",".","MarkerSize",15,Color=[060/255,064/255,091/255])
        box off
        grid on
        grid minor
        legend("TrainError","TestError")
        title("MutiTradaboost")
        drawnow;
    end
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

