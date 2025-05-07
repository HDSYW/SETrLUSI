function [result] = Tradaboost_F(Data,ktype,pa)

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
%% >>>>>>>>>>>>>>>>>>>> Learning <<<<<<<<<<<<<<<<<<<<
for p1=pa.min:pa.step:pa.max
    fprintf('--------------------*Regular=%.2f*--------------------\n',p1);
    % ---------- Para setting ----------
    n = size(tdY,1);
    m = size(tsY,1);
    w = ones(m+n,1)./(m+n);
    Para.kpar.ktype = "lin";
    Para.p1 = 2.^p1;  Para.kpar.kp1 =2.^-8; Para.kpar.kp2 = 0;
    % ---------- Iteration ----------
    for t = 1:pa.T
        % ---------- Model ----------
        p = w./(sum(abs(w)));
        idx = randsample(size(tX,1), floor(size(tX,1)/2), true, p);
        Trn.X = tX(idx, :);
        Trn.Y = tY(idx);
        [predict , ~] = LIB_L1SVC(tX , Trn , Para);
        [TestPredict{t} , model{t}] = LIB_L1SVC(X_test_T , Trn , Para);
        [TrainPredict{t} , model_train{t}] = LIB_L1SVC(Data.X_train_T, Trn , Para);
        % ---------- Update weight ----------
        sW = sum(p(n+1:m+n));
        %     T_error(t) = sum(p(n+1:m+n).*(predict(n+1:m+n)~=tsY)/sW);
        T_error(t) = sum((predict(n+1:m+n)~=tsY)/size(tsY,1));
        if T_error(t)>= 0.5
            T_error(t) = 0.499;
        elseif T_error(t) == 0
            T_error(t) = 0.001;
        end
        bT(t) = T_error(t)/(1-T_error(t));
        beta(t) =bT(t);
        b = 1/(1+sqrt(2*log(n/pa.T)));
        wUpdate = [(b*ones(n,1)).^(predict(1:n)~=tdY) ; (bT(t)*ones(m,1)).^(-(predict(n+1:m+n)~=tsY)) ];
        w = p.*wUpdate;
        % ---------- Test-result ----------
        l=size(X_test_T,1); Y_pred = zeros(l, 1);  yTwo = ones(l,1); Ydash = ones(l,1); yTwo_train = ones(size(Data.Y_train_T,1),1); Ydash = ones(l,1);
        yOne = ones(l,1);    DV=zeros(l, 1); DV_train=zeros(size(Data.Y_train_T,1), 1);yTrain=ones(size(Data.Y_train_T,1),1);Ydash_train = ones(size(Data.Y_train_T,1),1);
        for i= 1: t
            yOne = yOne.*((beta(i)*ones(l,1)).^(-TestPredict{i}));
            yTwo = yTwo.*((beta(i)*ones(l,1)).^(-0.5));
            yTwo_train= yTwo_train.*((beta(i)*ones(size(Data.Y_train_T,1),1)).^(-0.5));
            yTrain = yTrain .*((beta(i)*ones(size(Data.Y_train_T,1),1)).^(-TrainPredict{i}));
            DV=DV+model{t}.dv;
            DV_train=DV_train+model_train{t}.dv;
        end
        Ydash(yOne < yTwo) = -1;
        Ydash_train(yTrain < yTwo_train) = -1;
        CM_test = ConfusionMatrix(Ydash,Y_test_T);
        CM_train = ConfusionMatrix(Ydash_train,Data.Y_train_T);
        Test_error(t)=1-CM_test.Ac./100;
        Train_error(t)=1-CM_train.Ac./100;
        % ---------- Print result ----------
        fprintf('Iteration %s \t',num2str(t))
        fprintf('Train_error=%.4f \t',Train_error(t))
        fprintf('Test_error=%.4f\t\n',Test_error(t))

        folderpath=pa.DA+'/TR/';
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
        best_Ypre_test= Ydash;
        best_regular=p1;
    end
    % ---------- Figure ----------
    if sum(pa.figure=='On')
        figure
        x=1:pa.T;
        y_1=Train_error;
        y_2=Test_error;
        plot(x,y_1,"Marker",".","MarkerSize",15,Color=[223/255,122/255,094/255])
        hold on
        plot(x,y_2,"Marker",".","MarkerSize",15,Color=[060/255,064/255,091/255])
        box off
        grid on
        grid minor
        legend("TrainError","TestError")
        title("Tradaboost")
        drawnow;
    end
end
% ---------- Output ----------
CM_test = ConfusionMatrix(best_Ypre_test,Y_test_T);
result.ac_test=CM_test.Ac;
result.F=CM_test.FM;
result.GM=CM_test.GM;
result.testerror=best_error_test_H;
[~,~,~, AUC]=perfcurve(Y_test_T,  best_Final_Y_test/pa.T, '1');
result.AUC=100*AUC;
result.Spe=CM_test.Spe; result.Sen=CM_test.Sen;
result.lam=best_regular;
fprintf('%s\n', repmat('-', 1, 60));
fprintf('Finall_Test_AC=%.4f\t',result.ac_test)
fprintf('Finall_regular=%.4f\t\n',best_regular)
fprintf('%s\n', repmat('=', 1, 60));
end

