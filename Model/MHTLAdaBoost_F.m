function [result] = MHTLAdaBoost_F(Data,ktype,pa)

% ---------- Data process ----------
tdY=Data.Y_train_A;
tsY=Data.Y_train_T;
tX=[Data.X_train_A;Data.X_train_T];
tY=[Data.Y_train_A;Data.Y_train_T];
X_test_T=Data.X_test_T;
Y_test_T=Data.Y_test_T;
tY(tY==0)=-1;
tsY(tsY==0)=-1;
tdY(tdY==0)=-1;
Data.Y_train_A(Data.Y_train_A==0)=-1;

fprintf('--------------------*Cluster Start*--------------------\n');
% Data_new=Layer1_MHTL_AdaBoost([Data.X_train_A,Data.Y_train_A], X_test_T, pa.k);
for k=3
    Data_new{k}=Layer1_MHTL_AdaBoost([Data.X_train_A,Data.Y_train_A], X_test_T, k);
    fprintf('*K=%.0f*NS_Useful=%.0f*\n',k,size(Data_new{k},1));
    num(k)=size(Data_new{k},1);
end
[~,idx_max]=max(num);
Data_new=Data_new{idx_max};
% fprintf('*K=%.0f*NS_Useful=%.0f*\n',k,size(Data_new(:,end),1));
fprintf('--------------------*Cluster End*--------------------\n');
if size(Data_new,1)<=pa.T
    Data_new=[Data.X_train_A,Data.Y_train_A];
    tdY=Data.Y_train_A;
    fprintf('*No Similar Sample');
else
    tdY=Data_new(:,end);
end
tX=[Data_new(:,1:end-1);Data.X_train_T];
tY=[Data_new(:,end);Data.Y_train_T];
best_error_test=1;
%% >>>>>>>>>>>>>>>>>>>> Learning <<<<<<<<<<<<<<<<<<<<
for p1=pa.min:pa.step:pa.max
    fprintf('--------------------*Regular=%.2f*--------------------\n',p1);
    % ---------- Para setting ----------
    n = size(Data_new,1);
    m = size(tsY,1);
    w = ones(m+n,1)./(m+n);
    Para.kpar.ktype = "lin";
    Para.p1 = 2.^p1;  Para.kpar.kp1 =2.^-8; Para.kpar.kp2 = 0;
    % ---------- Iteration ----------
    for t = 1:pa.T
        % ---------- Model ----------
        p = w./(sum(abs(w)));
        idx = randsample(size(tX,1), floor(size(tX,1)./2), true, p);
        Trn.X = tX(idx, :);
        Trn.Y = tY(idx);
        [predict , ~] = LIB_L1SVC(tX , Trn , Para);
        [TestPredict{t} , model{t}] = LIB_L1SVC(X_test_T , Trn , Para);
        [TrainPredict{t} , model_train{t}] = LIB_L1SVC(Data.X_train_T, Trn , Para);
        % ---------- Update weight ----------
        sW = sum(p(n+1:m+n));
        T_error(t) = sum(p(n+1:m+n).*(predict(n+1:m+n)~=tY(n+1:m+n))/sW);
%         T_error(t) = sum((predict(n+1:m+n)~=tsY)/size(tsY,1));
        if T_error(t)>= 0.5
            T_error(t) = 0.499;
        elseif T_error(t) == 0
            T_error(t) = 0.001;
        end
        bT(t) = 0.5.*log((1-T_error(t))/(T_error(t)));
        beta(t) =bT(t);
        b = 0.5.*log(1+sqrt(2*log(n/pa.T)));
        wUpdate = [exp((-b*ones(n,1)).*(predict(1:n)~=tdY)) ; exp(bT(t)*ones(m,1).*(predict(n+1:m+n)~=tsY)) ];
        w = p.*wUpdate;
        % ---------- Test-result ----------
        l=size(X_test_T,1); Y_pred = zeros(l, 1);  yTwo = ones(l,1); Ydash = ones(l,1); yTwo_train = ones(size(Data.Y_train_T,1),1); Ydash = ones(l,1);
        yOne = ones(l,1);    DV=zeros(l, 1); DV_train=zeros(size(Data.Y_train_T,1), 1);yTrain=ones(size(Data.Y_train_T,1),1);Ydash_train = ones(size(Data.Y_train_T,1),1);
        for i= 1: t
            yOne = yOne+(beta(i).*(TestPredict{i}));
            DV=DV+beta(i).*model{t}.dv;
        end
       Ydash=sign(yOne);
        CM_test = ConfusionMatrix(Ydash,Y_test_T);
        Test_error(t)=1-CM_test.Ac./100;
        Train_error(t)=T_error(t);
        % ---------- Print result ----------
        fprintf('Iteration %s \t',num2str(t))
        fprintf('Train_error=%.4f \t',Train_error(t))
        fprintf('Test_error=%.4f\t\n',Test_error(t))

        folderpath=pa.DA+'/MHT/';
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
    clear bT  beta TrainPredict TestPredict Train_error Test_error
%     % ---------- Figure ----------
%     if sum(pa.figure=='On')
%         figure
%         x=1:pa.T;
%         y_1=Train_error;
%         y_2=Test_error;
%         plot(x,y_1,"Marker",".","MarkerSize",15,Color=[223/255,122/255,094/255])
%         hold on
%         plot(x,y_2,"Marker",".","MarkerSize",15,Color=[060/255,064/255,091/255])
%         box off
%         grid on
%         grid minor
%         legend("TrainError","TestError")
%         title("Tradaboost")
%         drawnow;
%     end
end
% ---------- Output ----------
CM_test = ConfusionMatrix(best_Ypre_test,Y_test_T);
result.ac_test=CM_test.Ac;
result.F=CM_test.FM;
result.GM=CM_test.GM;
result.testerror=best_error_test_H;
% [~,~,~, AUC]=perfcurve(Y_test_T,  sigmoid(best_Final_Y_test/pa.T), '1');
% result.AUC=100*AUC;
result.Spe=CM_test.Spe; result.Sen=CM_test.Sen;
result.lam=best_regular;
fprintf('%s\n', repmat('-', 1, 60));
fprintf('Finall_Test_AC=%.4f\t',result.ac_test)
fprintf('Finall_regular=%.4f\t\n',best_regular)
fprintf('%s\n', repmat('=', 1, 60));
end

