function [result] = LUSI_VSVM_F(Data,K,ktype,v_ker,CDFx,ptype,pa)
%test_ac=0;
test_GM=0;
X_train_T=Data.X_train_T;
Y_train_T=Data.Y_train_T;
X_test_T=Data.X_test_T;
Y_test_T=Data.Y_test_T;
k=K;
Y_train_T(Y_train_T==-1)=0;Y_test_T(Y_test_T==-1)=0;
Para.kpar.ktype = ktype;

V_Matrix = 'Vmatrix';
V_MatrixFun = str2func(V_Matrix);
Para.vmatrix = V_Matrix;
Para.CDFx = CDFx;
Para.v_ker = v_ker;
CDFx = Para.CDFx;

indices = crossvalind('Kfold',X_train_T(:,1),k);
for power2 = pa.min:pa.step:pa.max
    Para.kpar.kp1 =2.^power2;
    Para.kpar.kp2 = 0;
    for power1 = pa.min:pa.step:pa.max
        Para.v_sig = 2.^power1;
        for j = pa.min:pa.step:pa.max
            Para.p1=2.^j;
            for tao= pa.taomin:pa.taostep:pa.taomax
                Para.p3=tao;
                for i = 1:k
                    test = (indices == i); train = ~test;
                    Trn.X = X_train_T(train,:); Trn.Y = Y_train_T(train,:);
                    ValX = X_train_T(test,:); ValY = Y_train_T(test,:);
                    % ---------- Vmatrix ----------
                    [V,~] = Vmatrix(Trn.X,CDFx,Para.v_sig,Para.v_ker); Para.V=V;
                    Para.V=eye(size(Trn.X,1),size(Trn.X,1));
                    %  Para.P=ones(size(Trn.X,1),size(Trn.X,1));
                    Para.P=Pcaculate(Trn.X , Trn.Y,ptype);
                    % ---------- Model ----------
                    [PredictY , model] = LUSI_VSVM(ValX , Trn , Para); 
                    PredictY.tst(PredictY.tst==-1)=0;
                    M_Acc(i) = sum(PredictY.tst==ValY)/length(ValY)*100;
                    CM = ConfusionMatrix(PredictY.tst,ValY);        M_F(i)=CM.FM;
                    M_Erro(i) = sum(abs(model.prob - ValY));        M_Errorate(i) = sum(abs(model.prob - ValY))/sum(abs(ValY));
                    M_GM(i)=CM.GM;
                end
            end
            mean_GM=mean(M_GM) ;mean_Acc =mean(M_Acc); mean_F=mean(M_F); mean_Erro=mean(M_Erro); mean_Errorate=mean(M_Errorate);
            if  mean_GM>test_GM       % mean_Acc>test_ac or mean_F>test_F
                test_GM=mean_GM;         test_ac=mean_Acc;       best_Erro=mean_Erro;    best_Errorate=mean_Errorate;
                best_v_sig=Para.v_sig; best_kp1=Para.kpar.kp1; best_p1=Para.p1;best_p3=Para.p3;
            end
        end
    end
    fprintf('Complete %s\t\n',num2str((power2+8)*100/16))
end
% >>>>>>>>>>>>>>>>>>>> Test and prediction <<<<<<<<<<<<<<<<<<<<
Trn.X=X_train_T  ;    Trn.Y=Y_train_T;
Para.v_sig=best_v_sig   ;    Para.kpar.kp1=best_kp1; Para.p1=best_p1;Para.p3=best_p3;
% ---------- Vmatrix ----------
[V,~] = Vmatrix(Trn.X,CDFx,Para.v_sig,Para.v_ker); Para.V=V;
%                         Para.P=Pcaculate(Trn.X , Trn.Y,ptype);
Para.P=eye(size(Trn.X,1),size(Trn.X,1));
% ---------- Model ----------
[PredictY , ~] = LUSI_VSVM(X_test_T , Trn , Para);
PredictY.tst(PredictY.tst==-1)=0;
CM = ConfusionMatrix(PredictY.tst,Y_test_T) ;
result.ac_test=sum(PredictY.tst==Y_test_T)/length(Y_test_T)*100;
result.F=CM.FM;
result.GM=CM.GM;
result.Precision=CM.Precision;
result.Recall=CM.Recall;
[~,~,~, AUC]=perfcurve(Y_test_T, PredictY.tst, '1');
result.AUC=100*AUC;
result.lam=best_p1;
result.kp1=best_kp1;
result.v_sig=best_v_sig;
result.p4=best_p3;
fprintf('%s\n', repmat('-', 1, 100))        ; fprintf('Test_AC=%.2f||',result.ac_test);
fprintf('Train_GM=%.2f||\n',test_GM)     ; fprintf('Test_GM=%.2f||\n',result.GM);
fprintf('BestC=%.2f||',log2(best_p1))   ;  fprintf('Best_v_sig=%.2f||',log2(best_v_sig));
fprintf('Best_kp1=%.2f||',log2(best_kp1)) ; fprintf('Best_tao=%.2f||\n',best_p3) ; fprintf('%s\n', repmat('=', 1, 100));
end

