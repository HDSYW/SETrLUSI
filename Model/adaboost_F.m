function result= adaboost_F(Data,ktype,pa)
% ---------- Data process ----------
tdY=Data.Y_train_A;
tsY=Data.Y_train_T;

tX=[Data.X_train_A;Data.X_train_T];
tY=[Data.Y_train_A;Data.Y_train_T];
X_test_T=Data.X_test_T;
Y_test_T=Data.Y_test_T;
tY(tY==0)=-1;tY(tY==0)=-1;
tsY(tsY==0)=-1;tsY(tsY==0)=-1;
tdY(tdY==0)=-1;tdY(tdY==0)=-1;
best_error_test=1;

for p1=pa.min:pa.step:pa.max
    fprintf('-------------------------------------------*Regular=%.2f*--------------------------------------\n',p1);
    % ---------- Para setting ----------
    n = size(tdY,1);
    m = size(tsY,1);
    weights = ones(m+n,1)./(m+n);
    alphas = zeros(pa.T, 1);
    Para.p1 = 2^p1;
    Para.kpar.kp1 =2.^0;
    Para.kpar.kp2 = 0;
    Para.kpar.ktype = "lin";

    for t = 1:pa.T
        %     lib_opt =  sprintf('-t 2 -c %f -g %f -q', 2.^2, 2.^2);
        idx = randsample(size(tX,1), floor(size(tX,1)/2), true, weights);
        Trn.X = tX(idx, :);
        Trn.Y = tY(idx);
        weights = weights / sum(weights);
        [predict , ~] = LIB_L1SVC(tX , Trn , Para);
        [TestPredict{t} , model{t}] = LIB_L1SVC(X_test_T , Trn , Para);
        [TrainPredict{t} , model_train{t}] = LIB_L1SVC(tX , Trn , Para);
        % ---------- Update weight ----------
        err(t) = sum(weights.*(tY ~=  predict))./sum(weights);
        if err(t)>= 0.5
            err(t) = 0.499;
        elseif err(t) == 0
            err(t) = 0.001;
        end
        alpha(t) = 0.5 * log((1 - err(t)) / err(t));
        weights = weights .* exp(-alpha(t) * tY .* predict);

        alphas(t) = alpha(t);

        % ---------- Test-result ----------
        y_test_pred = zeros(size(X_test_T,1), 1);
        y_train_pred = zeros(size(tY,1), 1);
        for l = 1:t
            y_test_pred = y_test_pred +alphas(l) .* TestPredict{l};
            y_train_pred = y_train_pred +alphas(l) .* TrainPredict{l};
        end
        y_test_pred = sign(y_test_pred);
        y_train_pred = sign(y_train_pred);
        DV=model{t}.dv;
        test_error(t) = sum(y_test_pred ~= Y_test_T) / size(Y_test_T,1);
        train_error(t) = sum( y_train_pred ~= tY) / size(tY,1);
        fprintf('Iteration %s \t',num2str(t))
        fprintf('Train_error=%.4f \t',train_error(t))
        fprintf('Test_error=%.4f \t\n',test_error(t))

        folderpath=pa.DA+'/AD/';
        if t==1
            mkdir(folderpath);
        end
        filename=sprintf('file_%d.mat',p1);
        fullpath=fullfile(folderpath,filename);
        save(fullpath,'test_error')
    end
    if test_error(pa.T)<=best_error_test
        best_error_test_H=test_error;
        best_error_test= test_error(pa.T);
        best_Final_Y_test=DV;
        best_Ypre_test= y_test_pred;
        best_regular=p1;
    end
    % ---------- Figure ----------
    if sum(pa.figure=='On')
        figure
        x=1:pa.T;
        y_1=train_error;
        y_2=test_error;
        plot(x,y_1,"Marker",".","MarkerSize",15,Color=[223/255,122/255,094/255])
        hold on
        plot(x,y_2,"Marker",".","MarkerSize",15,Color=[060/255,064/255,091/255])
        box off
        grid on
        grid minor
        legend("TrainError","TestError")
        drawnow;
        title("AdaboostSVM")
    end
end
% ---------- Output ----------
CM_test = ConfusionMatrix(best_Ypre_test,Y_test_T);
result.ac_test=CM_test.Ac;
result.F=CM_test.FM;
result.GM=CM_test.GM;
result.testerror=best_error_test_H;
[~,~,~, AUC]=perfcurve(Y_test_T, sigmoid(best_Final_Y_test), '1');
result.AUC=100*AUC;
result.Spe=CM_test.Spe; result.Sen=CM_test.Sen;
result.lam=best_regular;
fprintf('%s\n', repmat('-', 1, 100));
fprintf('Finall_Test_AC=%.4f\t',result.ac_test)
fprintf('Finall_regular=%.4f\t\n',best_regular)
fprintf('%s\n', repmat('=', 1, 100));
end